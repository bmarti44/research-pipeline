"""
BPE Tokenizer for COCONUT A+B+C Study (v3.4)

Wraps tiktoken GPT-2 encoding and extends with COCONUT special tokens.
This allows proper subword tokenization to test COCONUT's mechanism -
compressing reasoning steps (subwords) rather than characters.

Usage:
    from bpe_tokenizer import BPETokenizer, get_tokenizer

    # Direct instantiation
    tokenizer = BPETokenizer()

    # Factory function (for train_abc.py)
    tokenizer = get_tokenizer("bpe")  # or "char" for SimpleTokenizer
"""

from typing import List, Union

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class BPETokenizer:
    """
    BPE tokenizer using tiktoken GPT-2 encoding with COCONUT special tokens.

    Vocab layout:
    - 0-50256: GPT-2 BPE tokens (50257 tokens)
    - 50257: <pad>
    - 50258: <eos>
    - 50259: <bot>      (beginning of thought)
    - 50260: <thought>  (latent thought placeholder)
    - 50261: <eot>      (end of thought)

    Total vocab_size: 50262
    """

    def __init__(self):
        if not TIKTOKEN_AVAILABLE:
            raise ImportError(
                "tiktoken is required for BPE tokenization. "
                "Install with: pip install tiktoken"
            )

        # Load GPT-2 encoding
        self._enc = tiktoken.get_encoding("gpt2")
        self._gpt2_vocab_size = self._enc.n_vocab  # 50257

        # Define special tokens (appended after GPT-2 vocab)
        self.special_tokens = {
            "<pad>": self._gpt2_vocab_size + 0,      # 50257
            "<eos>": self._gpt2_vocab_size + 1,      # 50258
            "<bot>": self._gpt2_vocab_size + 2,      # 50259
            "<thought>": self._gpt2_vocab_size + 3,  # 50260
            "<eot>": self._gpt2_vocab_size + 4,      # 50261
        }

        # Reverse mapping for decode
        self.id_to_special = {v: k for k, v in self.special_tokens.items()}

        # Total vocabulary size
        self.vocab_size = self._gpt2_vocab_size + len(self.special_tokens)  # 50262

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs using GPT-2 BPE.

        Special tokens in the text (e.g., "<thought>") are NOT recognized -
        use the token IDs directly for special tokens.
        """
        return self._enc.encode(text, allowed_special=set())

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text.

        Special tokens (>=50257) are converted to their string representations.
        """
        result = []
        regular_ids = []

        for token_id in ids:
            if token_id in self.id_to_special:
                # Flush regular tokens first
                if regular_ids:
                    result.append(self._enc.decode(regular_ids))
                    regular_ids = []
                # Skip special tokens in output (or optionally include them)
                # result.append(self.id_to_special[token_id])
            elif token_id < self._gpt2_vocab_size:
                regular_ids.append(token_id)
            # Ignore invalid token IDs >= vocab_size

        # Flush remaining regular tokens
        if regular_ids:
            result.append(self._enc.decode(regular_ids))

        return "".join(result)

    @property
    def pad_token_id(self) -> int:
        return self.special_tokens["<pad>"]

    @property
    def eos_token_id(self) -> int:
        return self.special_tokens["<eos>"]

    @property
    def bot_token_id(self) -> int:
        return self.special_tokens["<bot>"]

    @property
    def thought_token_id(self) -> int:
        return self.special_tokens["<thought>"]

    @property
    def eot_token_id(self) -> int:
        return self.special_tokens["<eot>"]


class SimpleTokenizer:
    """
    Character-level tokenizer for fair comparison.
    (Copied from train_abc.py for import convenience)
    """

    def __init__(self):
        # Special tokens
        self.special_tokens = {
            "<pad>": 0,
            "<eos>": 1,
            "<bot>": 2,      # Beginning of thought
            "<thought>": 3,  # Thought token (for COCONUT)
            "<eot>": 4,      # End of thought
        }

        # Build vocab (ASCII printable + special)
        self.char_to_idx = dict(self.special_tokens)
        for i in range(32, 127):  # Printable ASCII
            self.char_to_idx[chr(i)] = len(self.char_to_idx)

        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx.get(c, self.char_to_idx[" "]) for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.idx_to_char.get(i, "?") for i in ids)

    @property
    def pad_token_id(self) -> int:
        return self.special_tokens["<pad>"]

    @property
    def eos_token_id(self) -> int:
        return self.special_tokens["<eos>"]

    @property
    def thought_token_id(self) -> int:
        return self.special_tokens["<thought>"]

    @property
    def bot_token_id(self) -> int:
        return self.special_tokens["<bot>"]

    @property
    def eot_token_id(self) -> int:
        return self.special_tokens["<eot>"]


def get_tokenizer(tokenizer_type: str = "char") -> Union[SimpleTokenizer, BPETokenizer]:
    """
    Factory function to get tokenizer by type.

    Args:
        tokenizer_type: "char" for character-level, "bpe" for GPT-2 BPE

    Returns:
        Tokenizer instance with consistent interface
    """
    if tokenizer_type == "char":
        return SimpleTokenizer()
    elif tokenizer_type == "bpe":
        return BPETokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. Use 'char' or 'bpe'.")


if __name__ == "__main__":
    # Test both tokenizers
    print("=" * 60)
    print("BPE Tokenizer Test")
    print("=" * 60)

    # Test BPE tokenizer
    bpe = BPETokenizer()
    print(f"BPE vocab size: {bpe.vocab_size}")
    print(f"Special tokens: {bpe.special_tokens}")

    test_text = "John has 5 apples. Mary gives him 3 more. How many apples does John have?"
    encoded = bpe.encode(test_text)
    print(f"\nTest text: {test_text}")
    print(f"Encoded ({len(encoded)} tokens): {encoded[:20]}...")
    print(f"Decoded: {bpe.decode(encoded)}")

    # Test special tokens
    print(f"\nSpecial token IDs:")
    print(f"  <pad>: {bpe.pad_token_id}")
    print(f"  <eos>: {bpe.eos_token_id}")
    print(f"  <bot>: {bpe.bot_token_id}")
    print(f"  <thought>: {bpe.thought_token_id}")
    print(f"  <eot>: {bpe.eot_token_id}")

    # Test sequence with special tokens
    seq_with_special = encoded + [bpe.bot_token_id, bpe.thought_token_id, bpe.thought_token_id, bpe.eot_token_id]
    print(f"\nSequence with special tokens: {seq_with_special[-10:]}")
    print(f"Decoded (special tokens stripped): {bpe.decode(seq_with_special)}")

    print("\n" + "=" * 60)
    print("Character Tokenizer Test")
    print("=" * 60)

    # Test char tokenizer
    char = SimpleTokenizer()
    print(f"Char vocab size: {char.vocab_size}")

    char_encoded = char.encode(test_text)
    print(f"\nTest text: {test_text}")
    print(f"Encoded ({len(char_encoded)} tokens): {char_encoded[:20]}...")
    print(f"Decoded: {char.decode(char_encoded)}")

    print("\n" + "=" * 60)
    print("Compression Comparison")
    print("=" * 60)
    print(f"BPE tokens: {len(encoded)}")
    print(f"Char tokens: {len(char_encoded)}")
    print(f"Compression ratio: {len(char_encoded) / len(encoded):.1f}x")

    print("\n" + "=" * 60)
    print("Factory Function Test")
    print("=" * 60)

    for tok_type in ["char", "bpe"]:
        tok = get_tokenizer(tok_type)
        print(f"get_tokenizer('{tok_type}'): vocab_size={tok.vocab_size}, type={type(tok).__name__}")
