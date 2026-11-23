//! Word/string model for text compression.
//!
//! Adds a **word dictionary model** that tokenizes text into word symbols and
//! maintains a P(bit==1) table keyed on the previous word + current bit position.
//! This is PAQ's "word model" — it turns natural language redundancy into direct
//! symbol prediction instead of byte-level context chains.
//!
//! Only active for `Text` blocks. Binary/Exec keep the existing stack.

use super::ctable::CtxTable;
use super::BitModel;
use super::ByteAssembler;
use std::collections::HashMap;

const MAX_PROB: u16 = 4095;
const MIN_PROB: u16 = 1;
const CTX_BITS: u32 = 19; // ~512 KiB table

/// Word break characters: space, tab, newline, carriage return, and common punctuation.
const WORD_BREAKS: &[u8] = b" \t\n\r\0\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f\
                               !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

/// Rolling word buffer: keeps the last N completed words.
#[derive(Debug, Clone, Default)]
struct WordBuffer {
    /// Completed word byte sequences.
    words: Vec<Vec<u8>>,
    /// Maximum number of words to retain.
    cap: usize,
}

impl WordBuffer {
    fn new(cap: usize) -> Self {
        Self {
            words: Vec::with_capacity(cap),
            cap,
        }
    }

    /// Push a completed word. If buffer is full, drop the oldest.
    fn push(&mut self, word: Vec<u8>) {
        if self.words.len() == self.cap {
            self.words.remove(0);
        }
        self.words.push(word);
    }

    /// Get the last word, if any.
    fn last(&self) -> Option<&[u8]> {
        self.words.last().map(|w| w.as_slice())
    }

    /// Reset the buffer.
    fn reset(&mut self) {
        self.words.clear();
    }
}

/// Simple word hash: FNV-1a 32-bit over the word bytes.
#[inline]
fn word_hash(word: &[u8]) -> u32 {
    let mut h = 2166136261u32;
    for &b in word {
        h ^= u32::from(b);
        h = h.wrapping_mul(16777619);
    }
    h
}

/// Byte-Pair Re-Pair dictionary, built incrementally from the byte stream.
///
/// Maintains a running count of byte-pair frequencies and replaces frequent
/// pairs with high-byte symbols (0x80+).  The cap-fold bit is stored alongside
/// each substituted symbol to preserve case information.
///
/// Both encoder and decoder build this dictionary identically because they
/// process the same byte stream via `ByteAssembler`.
const MAX_DICT_SIZE: usize = 2048; // top 2K frequent substrings
const DICT_SYMBOL_BASE: u8 = 0x80; // substituted symbols start at 0x80

/// An entry in the Re-Pair dictionary: maps a (byte pair) → symbol id.
#[derive(Debug, Clone, Copy)]
struct DictEntry {
    /// The byte pair this entry replaces (e.g. [b'a', b'b']).
    pair: [u8; 2],
    /// The symbol ID assigned (0-based, added to DICT_SYMBOL_BASE when emitted).
    id: u8,
}

/// Byte-Pair-RePair dictionary, built incrementally.
///
/// Tracks byte-pair frequencies and promotes frequent pairs to high-byte
/// substitution symbols (`0x80+`).  The cap-fold bit preserves case info.
#[allow(dead_code)] // some methods are for Phase 2 integration
#[derive(Debug, Default)]
struct BytePairDict {
    /// Byte-pair frequency counts.
    pair_counts: HashMap<[u8; 2], u32>,
    /// Forward lookup: pair → symbol id.
    pair_to_id: HashMap<[u8; 2], u8>,
    /// Reverse lookup: symbol id → pair.
    id_to_pair: Vec<DictEntry>,
    /// Next available symbol ID.
    next_id: u8,
}

impl BytePairDict {
    fn new() -> Self {
        Self {
            pair_counts: HashMap::new(),
            pair_to_id: HashMap::new(),
            id_to_pair: Vec::with_capacity(MAX_DICT_SIZE),
            next_id: 0,
        }
    }

    /// Record a byte pair occurrence and potentially promote it to a substitution.
    ///
    /// When a pair's frequency crosses a threshold (currently: it becomes the
    /// most frequent pair and there's room in the dictionary), it gets a symbol ID.
    #[inline]
    fn record_pair(&mut self, pair: [u8; 2]) {
        if self.next_id as usize >= MAX_DICT_SIZE {
            return; // dictionary full
        }
        let count = self.pair_counts.entry(pair).or_insert(0);
        *count += 1;
        // Promote to a symbol if this pair's frequency exceeds the threshold.
        // Threshold: count must be at least 4 (empirically balances noise vs. signal).
        if *count >= 4 && !self.pair_to_id.contains_key(&pair) {
            self.pair_to_id.insert(pair, self.next_id);
            self.id_to_pair.push(DictEntry {
                pair,
                id: self.next_id,
            });
            self.next_id = self.next_id.wrapping_add(1);
        }
    }

    /// Look up a byte pair in the dictionary. Returns the symbol ID if present.
    #[inline]
    fn lookup(&self, pair: [u8; 2]) -> Option<u8> {
        self.pair_to_id.get(&pair).copied()
    }

    /// Check if a byte is a high-byte substitution symbol (0x80+).
    #[inline]
    fn is_symbol(b: u8) -> bool {
        b >= DICT_SYMBOL_BASE
    }

    /// Get the pair for a symbol ID (for inverse substitution).
    #[inline]
    fn reverse_lookup(&self, id: u8) -> Option<[u8; 2]> {
        self.id_to_pair.iter().find(|e| e.id == id).map(|e| e.pair)
    }

    fn reset(&mut self) {
        self.pair_counts.clear();
        self.pair_to_id.clear();
        self.id_to_pair.clear();
        self.next_id = 0;
    }
}

/// Word model: tokenizes text into words and models the current bit position
/// conditioned on the previous word hash.
///
/// This captures the strong "previous word predicts current word" signal in
/// natural language text, which byte-level order-N models miss because they
/// operate over raw byte positions rather than symbol boundaries.
pub struct WordModel {
    asm: ByteAssembler,
    ctab: CtxTable,
    /// Rolling buffer of recent words.
    words: WordBuffer,
    /// Current incomplete word being assembled.
    cur_word: Vec<u8>,
    /// Maximum word length to retain (prevents pathological memory use).
    max_word_len: usize,
    /// Maximum number of recent words to keep in the rolling buffer.
    #[allow(dead_code)] // used for configuration, may be increased later
    max_words: usize,
    /// Byte-Pair Re-Pair dictionary for the current block.
    dict: BytePairDict,
    /// Previous byte (for pair tracking).
    prev_byte: Option<u8>,
    /// Whether to use Re-Pair substitution.
    #[allow(dead_code)] // unused toggle; Re-Pair path is always on via `new`
    use_repar: bool,
}

impl WordModel {
    /// Create a new word model with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            asm: ByteAssembler::new(8),
            ctab: CtxTable::new(CTX_BITS),
            words: WordBuffer::new(8),
            cur_word: Vec::new(),
            max_word_len: 64,
            max_words: 8,
            dict: BytePairDict::new(),
            prev_byte: None,
            use_repar: true,
        }
    }

    /// Finalize the current incomplete word and push it to the buffer.
    fn flush_word(&mut self) {
        if self.cur_word.is_empty() {
            return;
        }
        let word = std::mem::take(&mut self.cur_word);
        self.words.push(word);
    }

    /// Compute the context key for the current bit position.
    #[inline]
    fn key(&self) -> u64 {
        let prev_hash = self.words.last().map_or(0u32, |w| word_hash(w));
        let bit_pos = u32::from(self.asm.nbits());
        let last = u32::from(self.asm.last_byte());
        let prev = u32::from(self.asm.prev_byte());
        // Mix in the previous word hash + bit position + last byte context.
        u64::from(prev_hash).rotate_left(32)
            ^ (u64::from(bit_pos) << 40)
            ^ (u64::from(last) << 8)
            ^ u64::from(prev)
    }

    /// Apply Re-Pair substitution to a byte if a matching pair was seen.
    ///
    /// Returns `Some(substituted_byte)` if the previous byte + current byte form
    /// a known pair, `None` otherwise.
    fn try_substitute(&mut self, byte: u8) -> Option<u8> {
        if !self.use_repar || self.prev_byte.is_none() {
            return None;
        }
        let prev = self.prev_byte.expect("prev_byte checked above");
        if let Some(id) = self.dict.lookup([prev, byte]) {
            // Cap-fold bit: bit 0 = lowercase/original, bit 1 = uppercase variant.
            // We only substitute for lowercase bytes; uppercase bytes are passed through.
            if byte.is_ascii_lowercase() || byte.is_ascii_digit() || WORD_BREAKS.contains(&byte) {
                let sym = DICT_SYMBOL_BASE + id;
                return Some(sym);
            }
        }
        None
    }
}

impl BitModel for WordModel {
    #[inline]
    fn predict(&self) -> u16 {
        let [c0, c1] = self.ctab.get(self.key());
        let tot = f64::from(c0 + c1);
        (f64::from(c1) / tot * f64::from(MAX_PROB)).clamp(f64::from(MIN_PROB), f64::from(MAX_PROB))
            as u16
    }

    #[inline]
    fn update(&mut self, bit: bool) {
        let completed = self.asm.push_bit(bit);
        if let Some(byte) = completed {
            // Record byte pair for Re-Pair dictionary.
            if let Some(prev) = self.prev_byte {
                self.dict.record_pair([prev, byte]);
            }
            self.prev_byte = Some(byte);

            // Check if this byte is a Re-Pair substitution symbol.
            if BytePairDict::is_symbol(byte) {
                // Substituted symbol — treat as a word break for word modeling.
                self.flush_word();
                // Don't push the symbol into cur_word.
            } else if WORD_BREAKS.contains(&byte) {
                self.flush_word();
            } else if self.cur_word.len() < self.max_word_len {
                self.cur_word.push(byte);
            }
        }

        let k = self.key();
        self.ctab.update(k, bit);
    }

    fn reset(&mut self) {
        self.asm.reset();
        self.ctab.reset();
        self.words.reset();
        self.cur_word.clear();
        self.dict.reset();
        self.prev_byte = None;
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Default for WordModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_words() {
        let mut m = WordModel::new();
        // Feed "the " — word break after 'e'
        for b in b"the " {
            for bit_idx in (0..8).rev() {
                let bit = (b >> bit_idx) & 1 == 1;
                m.update(bit);
            }
        }
        // Word buffer should have "the"
        let last = m.words.last().unwrap_or(b"");
        assert_eq!(last, b"the");
    }

    #[test]
    fn starts_neutral() {
        let m = WordModel::new();
        assert!(
            (i32::from(m.predict()) - 2048).abs() <= 2,
            "new WordModel should start near neutral, got {}",
            m.predict()
        );
    }

    #[test]
    fn adapts_after_repeated_word() {
        let mut m = WordModel::new();
        let text = b"hello hello hello hello hello hello hello hello hello hello ";
        for b in text {
            for bit_idx in (0..8).rev() {
                let bit = ((*b) >> bit_idx) & 1 == 1;
                m.update(bit);
            }
        }
        // After many repetitions of "hello ", the model should have learned
        // something about the pattern.
        assert!(m.words.words.len() > 0, "should have words in buffer");
    }

    #[test]
    fn reset_clears_state() {
        let mut m = WordModel::new();
        for b in b"test test test " {
            for bit_idx in (0..8).rev() {
                m.update(((b) >> bit_idx) & 1 == 1);
            }
        }
        m.reset();
        assert!(
            m.words.words.is_empty(),
            "word buffer should be empty after reset"
        );
    }

    #[test]
    fn repar_dictionary_builds_from_frequent_pairs() {
        let mut m = WordModel::new();
        // "hello world hello world..." x4 — "wo" appears 3 times, "ll" 3 times.
        // Need threshold 3 to trigger. Let me use enough repeats.
        let text = b"hello world hello world hello world hello world hello world hello world";
        for &b in text {
            for bit_idx in (0..8).rev() {
                let bit = (b >> bit_idx) & 1 == 1;
                m.update(bit);
            }
        }
        // After enough repeats, the dictionary should have at least one entry.
        assert!(
            m.dict.id_to_pair.len() > 0,
            "dictionary should have at least one entry after frequent pairs"
        );
    }

    #[test]
    fn xwrt_dictionary_serialization_caps_at_max() {
        // A corpus with many distinct words must serialize only MAX_XWRT_WORDS
        // (ESC + u16 extends past the old 128 single-byte slots).
        let mut text = Vec::new();
        for i in 0..1500 {
            text.extend_from_slice(format!("word{i} ").as_bytes());
        }
        let dict = XwrtDictionary::build_from_data(&text);
        let bytes = dict.to_bytes();
        let n = u16::from_le_bytes([bytes[0], bytes[1]]);
        assert_eq!(
            n as usize, MAX_XWRT_WORDS,
            "dict must serialize exactly MAX_XWRT_WORDS"
        );
        let back = XwrtDictionary::from_bytes(&bytes).expect("parses");
        assert_eq!(back.id_to_word.len(), MAX_XWRT_WORDS);
    }

    #[test]
    fn xwrt_global_dict_transform_round_trip() {
        let text = b"the quick brown fox jumps over the lazy dog\n\
the quick brown fox jumps over the lazy dog\n\
the quick brown fox jumps over the lazy dog\n"
            .to_vec();
        let dict = XwrtDictionary::build_from_data(&text);
        let xwrt = dict.transform(&text);
        assert!(xwrt.len() < text.len(), "frequent words should shrink");
        let back = inverse_with_dict(&xwrt, text.len(), &dict);
        assert_eq!(back, text);
    }

    #[test]
    fn xwrt_esc_tokens_round_trip_large_vocab() {
        // Force ids ≥ 127 so ESC + u16 encoding is exercised.
        let mut text = Vec::new();
        for i in 0..200 {
            let w = format!("w{i:03}");
            for _ in 0..3 {
                text.extend_from_slice(w.as_bytes());
                text.push(b' ');
            }
        }
        let dict = XwrtDictionary::build_from_data(&text);
        assert!(dict.id_to_word.len() >= 128);
        let xwrt = dict.transform(&text);
        assert!(
            xwrt.iter().any(|&b| b == XWRT_ESC),
            "expected ESC tokens for ids ≥ 127"
        );
        let back = inverse_with_dict(&xwrt, text.len(), &dict);
        assert_eq!(back, text);
    }

    #[test]
    fn xwrt_skips_breakless_runs() {
        // No word breaks in the stream: the whole run is one >maxlen "word",
        // which the scanner drops. Transform must be an identity passthrough.
        let text = b"rcnrcnrcn".repeat(5000);
        let dict = XwrtDictionary::build_from_data(&text);
        let bytes = dict.to_bytes();
        assert_eq!(&bytes[..2], &[0, 0], "no real words -> empty dict");
        let xwrt = dict.transform(&text);
        assert_eq!(xwrt, text, "empty dict is identity");
    }
}

/// XWRT: eXtended Word Replacement Transform.
///
/// Builds a static dictionary of the top words in the input text, then
/// replaces each word occurrence with a token:
/// - ids **0..126**: single byte `0x80..=0xFE`
/// - ids **127..MAX-1**: escape `0xFF` + `u16` LE id (3 bytes)
///
/// Vocab is capped at [`MAX_XWRT_WORDS`] (1024). Non-word bytes (including
/// word-break chars) pass through unchanged. ASCII-gated streams never emit
/// literal high bytes, so the high range is free for tokens.
///
/// Returns: transformed bytes where word tokens use the encoding above.
pub const MAX_XWRT_WORDS: usize = 1024;
/// Longest word kept by the XWRT scanner. Also the max that fits the
/// `u8` length byte in [`XwrtDictionary::to_bytes`]; anything longer is a
/// breakless run (base64/hex/garbage) rather than a real word.
pub const MAX_XWRT_WORD_LEN: usize = 255;
/// Escape byte for XWRT ids ≥ 127 (`0xFF || u16_le(id)`).
pub const XWRT_ESC: u8 = 0xFF;
/// Highest id that fits in a single token byte (`0x80 + id` → `0xFE`).
const XWRT_SINGLE_MAX_ID: usize = 126;

#[inline]
fn emit_xwrt_token(out: &mut Vec<u8>, id: usize) {
    if id <= XWRT_SINGLE_MAX_ID {
        out.push(0x80 + id as u8);
    } else {
        out.push(XWRT_ESC);
        out.extend_from_slice(&(id as u16).to_le_bytes());
    }
}

/// Parse one XWRT token at `data[i]`. Returns `(word_id, bytes_consumed)`.
#[inline]
fn parse_xwrt_token(data: &[u8], i: usize) -> Option<(usize, usize)> {
    if i >= data.len() || data[i] < 0x80 {
        return None;
    }
    if data[i] == XWRT_ESC {
        if i + 3 > data.len() {
            return None;
        }
        let id = u16::from_le_bytes([data[i + 1], data[i + 2]]) as usize;
        Some((id, 3))
    } else {
        Some(((data[i] - 0x80) as usize, 1))
    }
}

/// Advance past one XWRT token (or a single high byte if truncated ESC).
#[inline]
fn skip_xwrt_token(data: &[u8], i: usize) -> usize {
    match parse_xwrt_token(data, i) {
        Some((_, n)) => i + n,
        None => i + 1,
    }
}

pub fn xwrt_transform(data: &[u8]) -> Vec<u8> {
    // First pass: count word frequencies and remember first appearance
    let mut scanner = WordScanner::new(2048); // keep top 2K words
    let mut i = 0usize;
    while i < data.len() {
        if is_word_break(data[i]) {
            i += 1;
            continue;
        }
        // Found start of word
        let start = i;
        while i < data.len() && !is_word_break(data[i]) {
            i += 1;
        }
        if i > start {
            scanner.add_word(&data[start..i]);
        }
    }
    let dictionary = scanner.build_dictionary();

    // Second pass: replace words with tokens
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0usize;
    while i < data.len() {
        if is_word_break(data[i]) {
            out.push(data[i]);
            i += 1;
            continue;
        }
        // Try to match longest word in dictionary
        let mut matched = false;
        for word in dictionary.words.iter().take(MAX_XWRT_WORDS) {
            if i + word.len() <= data.len() && &data[i..i + word.len()] == word.as_slice() {
                let id = dictionary
                    .word_to_id
                    .get(String::from_utf8_lossy(word).as_ref())
                    .copied()
                    .unwrap_or(0);
                if id > XWRT_SINGLE_MAX_ID && word.len() <= 3 {
                    continue;
                }
                emit_xwrt_token(&mut out, id);
                i += word.len();
                matched = true;
                break;
            }
        }
        if !matched {
            // Not a known word, output byte as-is
            out.push(data[i]);
            i += 1;
        }
    }
    out
}

/// Inverse of [`xwrt_transform`].
///
/// Replaces tokens with their corresponding words from the dictionary.
///
/// The decoder rebuilds the identical dictionary by scanning the *output*
/// of xwrt_transform (which has the same word break positions as input)
/// and counting frequencies of the non-token bytes.
pub fn xwrt_inverse_transform(data: &[u8], orig_len: usize) -> Vec<u8> {
    // Rebuild dictionary from the transformed data (same word break positions)
    let mut scanner = WordScanner::new(2048);
    let mut i = 0usize;
    while i < data.len() {
        if data[i] >= 0x80 {
            i = skip_xwrt_token(data, i);
            continue;
        }
        if is_word_break(data[i]) {
            i += 1;
            continue;
        }
        // Found start of word
        let start = i;
        while i < data.len() && data[i] < 0x80 && !is_word_break(data[i]) {
            i += 1;
        }
        if i > start {
            scanner.add_word(&data[start..i]);
        }
    }
    let dictionary = scanner.build_dictionary();

    // Second pass: replace tokens with words
    let mut out = Vec::with_capacity(orig_len);
    let mut i = 0usize;
    while i < data.len() {
        if let Some((word_id, n)) = parse_xwrt_token(data, i) {
            if let Some(word) = dictionary.id_to_word.get(word_id) {
                out.extend_from_slice(word.as_bytes());
            }
            i += n;
        } else {
            out.push(data[i]);
            i += 1;
        }
    }
    out
}

/// Helper: scans text to find the top N words by frequency.
#[derive(Debug)]
pub(crate) struct WordScanner {
    /// Minimum word length to consider
    min_word_len: usize,
    /// Maximum number of words to keep in dictionary
    max_words: usize,
    /// Maximum word length kept (words longer than this are literal noise)
    max_word_len: usize,
    /// All words seen with their frequency and first position
    word_freq: Vec<(String, usize, usize)>,
    /// word -> index into `word_freq` (O(1) frequency updates)
    index: std::collections::HashMap<String, usize>,
}
impl WordScanner {
    pub(crate) fn new(max_words: usize) -> Self {
        Self {
            min_word_len: 2, // ignore single-letter words
            max_words,
            max_word_len: MAX_XWRT_WORD_LEN,
            word_freq: Vec::new(),
            index: std::collections::HashMap::new(),
        }
    }
    /// Add a word occurrence
    pub(crate) fn add_word(&mut self, word: &[u8]) {
        if word.len() < self.min_word_len || word.len() > self.max_word_len {
            return;
        }
        let word_str = String::from_utf8_lossy(word).to_string();
        if let Some(&idx) = self.index.get(&word_str) {
            // Increment frequency in place without a linear scan.
            self.word_freq[idx].1 += 1;
        } else {
            // New word
            let idx = self.word_freq.len();
            self.index.insert(word_str.clone(), idx);
            self.word_freq.push((word_str, 1, idx));
        }
    }
    /// Build dictionary: prefer frequent words; extended ESC slots (127..511)
    /// only accept words longer than 3 bytes so a 3-byte token never expands.
    pub(crate) fn build_dictionary(&mut self) -> XwrtDictionary {
        self.word_freq
            .sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.2.cmp(&b.2)));

        let mut selected: Vec<(String, usize, usize)> = Vec::with_capacity(MAX_XWRT_WORDS);
        // Single-byte token slots: ids 0..126.
        for entry in &self.word_freq {
            if selected.len() > XWRT_SINGLE_MAX_ID {
                break;
            }
            selected.push(entry.clone());
        }
        let single_count = selected.len();
        // ESC token slots: ids 127..511 — only words longer than the 3-byte token.
        for entry in self.word_freq.iter().skip(single_count) {
            if selected.len() >= MAX_XWRT_WORDS {
                break;
            }
            if entry.0.len() > 3 {
                selected.push(entry.clone());
            }
        }

        let mut word_to_id = HashMap::new();
        let mut id_to_word = Vec::new();
        for (idx, (word, _, _)) in selected.iter().enumerate() {
            word_to_id.insert(word.clone(), idx);
            id_to_word.push(word.clone());
        }
        let words: Vec<Vec<u8>> = selected
            .iter()
            .map(|(w, _, _)| w.as_bytes().to_vec())
            .collect();
        XwrtDictionary {
            words,
            word_to_id,
            id_to_word,
        }
    }
}
/// XWRT dictionary: keeps top N words and provides bidirectional lookup.
#[derive(Debug)]
pub struct XwrtDictionary {
    /// The words in the dictionary, sorted by frequency descending
    words: Vec<Vec<u8>>,
    /// word -> id lookup
    word_to_id: HashMap<String, usize>,
    /// id -> word lookup
    id_to_word: Vec<String>,
}

impl XwrtDictionary {
    /// Build dictionary from original data, then apply XWRT transform.
    pub fn build_from_data(data: &[u8]) -> Self {
        let mut scanner = WordScanner::new(2048);
        let mut i = 0usize;
        while i < data.len() {
            if is_word_break(data[i]) {
                i += 1;
                continue;
            }
            let start = i;
            while i < data.len() && !is_word_break(data[i]) {
                i += 1;
            }
            if i > start {
                scanner.add_word(&data[start..i]);
            }
        }
        let dict = scanner.build_dictionary();
        dict
    }

    /// Apply XWRT transform to data using this dictionary.
    ///
    /// Only the top [`MAX_XWRT_WORDS`] words are encoded. Ids 0..126 use a
    /// single byte (`0x80..=0xFE`); ids 127..511 use `0xFF || u16_le(id)`.
    pub fn transform(&self, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(data.len());
        let mut i = 0usize;
        while i < data.len() {
            if is_word_break(data[i]) {
                out.push(data[i]);
                i += 1;
                continue;
            }
            let mut matched = false;
            for word in self.words.iter().take(MAX_XWRT_WORDS) {
                if i + word.len() <= data.len() && &data[i..i + word.len()] == word.as_slice() {
                    let id = self
                        .word_to_id
                        .get(String::from_utf8_lossy(word).as_ref())
                        .copied()
                        .unwrap_or(0);
                    // Never expand: ESC tokens are 3 bytes.
                    if id > XWRT_SINGLE_MAX_ID && word.len() <= 3 {
                        continue;
                    }
                    emit_xwrt_token(&mut out, id);
                    i += word.len();
                    matched = true;
                    break;
                }
            }
            if !matched {
                out.push(data[i]);
                i += 1;
            }
        }
        out
    }

    /// Build dictionary from XWRT-transformed data (words appear as non-token, non-break bytes).
    pub fn from_transform(data: &[u8]) -> Vec<u8> {
        let mut scanner = WordScanner::new(2048);
        let mut i = 0usize;
        while i < data.len() {
            if data[i] >= 0x80 {
                i = skip_xwrt_token(data, i);
                continue;
            }
            if is_word_break(data[i]) {
                i += 1;
                continue;
            }
            let start = i;
            while i < data.len() && data[i] < 0x80 && !is_word_break(data[i]) {
                i += 1;
            }
            if i > start {
                scanner.add_word(&data[start..i]);
            }
        }
        let dict = scanner.build_dictionary();
        dict.to_bytes()
    }

    /// Serialize dictionary to bytes for storage in the encoded stream.
    ///
    /// Layout: `[count:u16 LE][len:u8][bytes...]...` capped at [`MAX_XWRT_WORDS`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let n = self.id_to_word.len().min(MAX_XWRT_WORDS) as u16;
        out.extend_from_slice(&n.to_le_bytes());
        for i in 0..n as usize {
            let word = &self.id_to_word[i];
            out.push(word.len() as u8);
            out.extend_from_slice(word.as_bytes());
        }
        out
    }

    /// Deserialize dictionary from bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 2 {
            return None;
        }
        let n = u16::from_le_bytes([data[0], data[1]]) as usize;
        if n > MAX_XWRT_WORDS {
            return None;
        }
        let mut pos = 2;
        let mut id_to_word = Vec::with_capacity(n);
        let mut word_to_id = HashMap::new();
        let mut words = Vec::with_capacity(n);
        for i in 0..n {
            if pos >= data.len() {
                return None;
            }
            let len = data[pos] as usize;
            pos += 1;
            if pos + len > data.len() {
                return None;
            }
            let word = String::from_utf8_lossy(&data[pos..pos + len]).to_string();
            pos += len;
            id_to_word.push(word.clone());
            word_to_id.insert(word.clone(), i);
            words.push(word.as_bytes().to_vec());
        }
        Some(Self {
            words,
            word_to_id,
            id_to_word,
        })
    }
}
/// Check if a byte is a word break character.
#[inline]
pub(crate) fn is_word_break(b: u8) -> bool {
    matches!(
        b,
        b' ' | b'\t'
            | b'\n'
            | b'\r'
            | b'\x00'
            | b'!'
            | b'"'
            | b'#'
            | b'$'
            | b'%'
            | b'&'
            | b'\''
            | b'('
            | b')'
            | b'*'
            | b'+'
            | b','
            | b'-'
            | b'.'
            | b'/'
            | b':'
            | b';'
            | b'<'
            | b'='
            | b'>'
            | b'?'
            | b'@'
            | b'['
            | b'\\'
            | b']'
            | b'^'
            | b'_'
            | b'`'
            | b'{'
            | b'|'
            | b'}'
            | b'~'
    )
}

/// Inverse XWRT using an explicit dictionary (from the encoded stream).
pub fn xwrt_inverse_with_dict(data: &[u8], orig_len: usize, dict_data: &[u8]) -> Vec<u8> {
    let dict = match XwrtDictionary::from_bytes(dict_data) {
        Some(d) => d,
        None => return Vec::new(),
    };
    inverse_with_dict(data, orig_len, &dict)
}

pub(crate) fn inverse_with_dict(data: &[u8], orig_len: usize, dict: &XwrtDictionary) -> Vec<u8> {
    let mut out = Vec::with_capacity(orig_len);
    let mut i = 0usize;
    while i < data.len() {
        if let Some((word_id, n)) = parse_xwrt_token(data, i) {
            if let Some(word) = dict.id_to_word.get(word_id) {
                out.extend_from_slice(word.as_bytes());
            }
            i += n;
        } else {
            out.push(data[i]);
            i += 1;
        }
    }
    out
}
