import re
from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from typing import Iterable
from transformers import AutoTokenizer


regex_whitespace = re.compile(r'\s')
CHUNK_SIZE = 2048 - 2
MAX_CHARS = 10_000
def create_text_chunks(text: str, tokenizer):

    def trim_to_whitespace(text: str, trim_start: bool = True, trim_end: bool = True):
        if trim_start:
            match = regex_whitespace.search(text)
            if match:
                text = text[match.start()+1:]
            else:
                text = text[10:]
        if trim_end:
            match = regex_whitespace.search(text[::-1])
            if match:
                text = text[:len(text) - match.start() - 1]
            else:
                text = text[:-10]
        return text


    # First tokenize the text
    # Speed hack, we take at most
    if len(text) <= 2*MAX_CHARS:
        tokens = tokenizer.encode(text[:MAX_CHARS], return_tensors="np", add_special_tokens=False)[0]
        # Process the top chunks
        chunks_from_top_sampled = [tokens[:CHUNK_SIZE]]

        chunks_top_text = tokenizer.batch_decode(chunks_from_top_sampled, skip_special_tokens=True)

        chunks_top_text = [trim_to_whitespace(chunks_top_text[0], trim_start=False, trim_end=True)]
        return chunks_top_text, []

    else:
        # We tokenize the top and bottom of text
        text_top = text[:MAX_CHARS]
        text_bottom = text[-MAX_CHARS:]

        tokens = tokenizer.batch_encode_plus([text_top, text_bottom], return_tensors="np", add_special_tokens=False)["input_ids"]

        # This ensures that the second chunks is always maxed out
        chunks = [tokens[0][:CHUNK_SIZE], tokens[1][-CHUNK_SIZE:]]

        chunks_text = tokenizer.batch_decode(chunks, skip_special_tokens=True)
        chunks_top_text = [trim_to_whitespace(chunks_text[0], trim_start=False, trim_end=True)]
        chunks_bottom_text = [trim_to_whitespace(chunks_text[1], trim_start=True, trim_end=False)]
        return chunks_top_text, chunks_bottom_text


class AddTextChunks(PipelineStep):
    name: str = "AddTextChunks"
    type: str = "Formatter"

    def __init__(self, tokenizer_name: str):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1) -> Iterable[Document]:
        for document in data:
            text = document.text
            chunks, _ = create_text_chunks(text, self.tokenizer)
            document.metadata["chunks"] = chunks
            yield document