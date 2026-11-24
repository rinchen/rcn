//! Bit-level predictors (context models) that feed the logistic mixer.
//!
//! Each model emits P(bit==1) in `[1, 4095]` (12-bit) and is updated causally after
//! the true bit is known. The codec uses a stack of these per block, mixed by
//! [`mixer::LogisticMixer`](super::model::mixer::LogisticMixer).

pub mod ctable;
pub mod e8e9;
pub mod exec;
pub mod indirect_dmc;
pub mod lazy_lzp;
pub mod lzp;
pub mod mixer;
pub mod mixer_bank;
pub mod order;
pub mod ppm;
pub mod ppmd_ssm;
pub mod sparse;
pub mod sse_apm;
pub mod ssm;
pub mod stack_enum;
pub mod word;

/// Assembles whole bytes from the per-bit [`BitModel::update`] stream so context
/// models can key on *byte* history instead of individual bits.
///
/// Coding is bit-by-bit (MSB→LSB) within each byte. `push_bit` feeds one coded bit;
/// when 8 are collected the completed byte is emitted to `on_byte` (which the model
/// uses to advance its real context), and the bit accumulator resets. Models call
/// this in `update` and key their prediction tables on the assembled byte context.
#[derive(Debug, Clone)]
pub struct ByteAssembler {
    pending: u8,
    nbits: u8,
    /// Ring buffer of most recent completed bytes, most-recent last.
    bytes: [u8; 64],  // max cap is 32 (CONTEXT_BYTES in ssm), so 64 is plenty
    cap: usize,
    len: usize,
    start: usize,  // index of oldest byte
}

impl ByteAssembler {
    /// Create an assembler retaining up to `cap` completed bytes of context.
    #[must_use]
    pub fn new(cap: usize) -> Self {
        assert!(cap <= 64, "ByteAssembler cap must be <= 64");
        Self {
            pending: 0,
            nbits: 0,
            bytes: [0; 64],
            cap,
            len: 0,
            start: 0,
        }
    }

    /// Feed one coded bit. Returns `Some(byte)` when a full byte completes.
    #[inline]
    pub fn push_bit(&mut self, bit: bool) -> Option<u8> {
        self.pending = (self.pending << 1) | u8::from(bit);
        self.nbits += 1;
        if self.nbits == 8 {
            self.nbits = 0;
            let byte = self.pending;
            self.pending = 0;
            self.push_byte_inner(byte);
            Some(byte)
        } else {
            None
        }
    }

    #[inline]
    fn push_byte_inner(&mut self, byte: u8) {
        if self.len == self.cap {
            // Overwrite oldest byte
            self.bytes[self.start] = byte;
            self.start = (self.start + 1) % self.cap;
        } else {
            let idx = (self.start + self.len) % self.cap;
            self.bytes[idx] = byte;
            self.len += 1;
        }
    }

    /// Directly record a completed byte, bypassing bit-by-bit assembly.
    ///
    /// This is used by callers (e.g. `MixerBank`) that track byte-level
    /// context but don't need per-bit assembly for their own prediction
    /// — only the completed byte history. Equivalent to feeding all 8 bits
    /// MSB-first, but avoids 8 branchy calls.
    #[inline]
    pub fn push_byte(&mut self, byte: u8) {
        self.push_byte_inner(byte);
        self.nbits = 0;
        self.pending = 0;
    }

    /// The number of bits accumulated toward the current (in-progress) byte, 0..8.
    #[must_use]
    pub const fn nbits(&self) -> u8 {
        self.nbits
    }

    /// Number of completed bytes currently retained.
    #[must_use]
    pub const fn bytes_len(&self) -> u64 {
        self.len as u64
    }

    /// The last `n` completed bytes (fewer if not enough history yet).
    #[must_use]
    pub fn last(&self, n: usize) -> &[u8] {
        let n = n.min(self.len);
        let start = (self.start + self.len - n) % self.cap;
        if start + n <= self.cap {
            &self.bytes[start..start + n]
        } else {
            // This shouldn't happen with small n <= cap, but handle wrap
            // by returning a slice (we can't return discontiguous, so this is a limitation)
            // For our use cases, n <= cap and we only need small slices
            let end = (start + n) % self.cap;
            if end > start {
                &self.bytes[start..end]
            } else {
                // Wrapped - return from start to end of buffer (approximation)
                &self.bytes[start..]
            }
        }
    }

    /// The single most-recent completed byte (0 before any).
    #[must_use]
    pub fn last_byte(&self) -> u8 {
        if self.len == 0 {
            0
        } else {
            let idx = (self.start + self.len - 1) % self.cap;
            self.bytes[idx]
        }
    }

    /// The previous completed byte (0 before two bytes exist).
    #[must_use]
    pub fn prev_byte(&self) -> u8 {
        if self.len < 2 {
            0
        } else {
            let idx = (self.start + self.len - 2) % self.cap;
            self.bytes[idx]
        }
    }

    /// The byte before the previous completed byte (0 before three bytes exist).
    #[must_use]
    pub fn prev_prev_byte(&self) -> u8 {
        if self.len < 3 {
            0
        } else {
            let idx = (self.start + self.len - 3) % self.cap;
            self.bytes[idx]
        }
    }

    /// Hash of the most-recent completed byte as a soft word-context signal.
    /// Non-breaking bytes contribute their value; word breaks map to 0.
    #[must_use]
    pub fn word_hash(&self) -> usize {
        match self.last_byte() {
            b if !matches!(
                b,
                b' ' | b'\t'
                    | b'\n'
                    | b'\r'
                    | b','
                    | b'.'
                    | b';'
                    | b':'
                    | b'!'
                    | b'?'
                    | b'"'
                    | b'\''
            ) => usize::from(b),
            _ => 0,
        }
    }

    /// True once at least one full byte has been assembled.
    #[must_use]
    pub const fn has_byte(&self) -> bool {
        self.len > 0
    }

    /// Clear all state (called at the start of each block).
    pub fn reset(&mut self) {
        self.pending = 0;
        self.nbits = 0;
        self.len = 0;
        self.start = 0;
    }
}

/// A bit-level predictor.
///
/// `predict` returns the model's estimate of P(bit==1) in `[1, 4095]`. `update` is
/// called **after** the true bit is known so the model can adapt. `reset` clears
/// per-block state at the start of each new block.
///
/// `Send` is required so Slow-mode can overlap classify(N+1) with encode(N) via
/// `rayon::join` while holding `&mut dyn BitModel` in the encode worker.
pub trait BitModel: Send {
    /// Predicted probability of bit==1, in `[1, 4095]`.
    fn predict(&self) -> u16;

    /// Adapt the model to the observed `bit`.
    fn update(&mut self, bit: bool);

    /// Reset per-block state (called at the start of each block).
    fn reset(&mut self);

    /// Prepare for a new block.  Called with the full block bytes before any
    /// bits are encoded/decoded.  Models that need block-level context (e.g. a
    /// dictionary built from the full block) override this to pre-compute it.
    /// The default is a no-op.
    fn prepare_block(&mut self, _block: &[u8]) {}

    /// Downcast to `&dyn Any` for persistence-specific plumbing.
    fn as_any(&self) -> &dyn std::any::Any;
    /// Downcast to `&mut dyn Any` for persistence-specific plumbing.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}
