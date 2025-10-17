import argparse
from pathlib import Path
import os
import json
import subprocess
from datatrove.pipeline.readers.base import BaseReader

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from datatrove.executor import LocalPipelineExecutor

parser = argparse.ArgumentParser("Quickly launch thom's style of tokenization.")

parser.add_argument(
    "data_paths", type=str, help="Path to the data to tokenize."
)
parser.add_argument(
    "--limit", type=int, help="limit the number of documents to tokenize", default=None
)
parser.add_argument(
    "--n_tasks", type=int, help="nb of tokenization tasks", default=100
)
parser.add_argument(
    "--max_toks", type=int, help="max tokens per file", default=1e8
)
# For avg 100k tokens we can set batch size to 2k for 8cpus with 2gb per cpu
parser.add_argument(
    "--batch_size", type=int, help="batch size", default=2000
)
parser.add_argument(
    "--qos", type=str, default="normal"
)
parser.add_argument(
    "--tokenizer", type=str, help="tokenizer to use", default="hynky/Llama-3.2-1B-no-bos"
)
parser.add_argument(
    "--text_key", type=str, default="text"
)
parser.add_argument(
    "--reader", type=str, choices=["jsonl", "parquet"], default="jsonl",
    help="Dataset reader to use. Defaults to 'jsonl'. Use 'parquet' for parquet inputs."
)
parser.add_argument(
    "--glob_pattern", type=str, default=None,
    help="Optional glob pattern to filter input files within the data folder"
)
parser.add_argument(
    "--sample", type=float, default=1.0
)
parser.add_argument(
    "--dep_job_id", type=str, default=None, help="ID of the job that produced the data to tokenize."
)
parser.add_argument(
    "--jsonl_output", "-jo", type=str, default=None, help="Path to optionally save the sampled data jsonl"
)
parser.add_argument(
    "--shuffle_chunk_size", "-scs", type=int, default=4096, help="Shuffle inter document"
)
parser.add_argument(
    "--run_merger", "-rm", action="store_true", help="Run the merger after tokenization"
)
parser.add_argument(
    "--name", "-n", type=str, default=None, help="Name of the tokenization"
)
parser.add_argument(
    "--max_chars_per_document", type=int, default=100_000, help="Split documents larger than this many characters"
)
parser.add_argument(
    "--enable_docling_postprocessing", action="store_true", help="Enable docling postprocessing to remove docling tags"
)
parser.add_argument(
    "--duplicate", type=int, default=1, help="Duplicate the data n times"
)
parser.add_argument(
    "--enable_normalize", action="store_true", help="Enable normalize postprocessor to truncate multiple newlines to single newlines"
)

class DocumentSplitter(PipelineStep):
    def __init__(self, max_chars_per_document: int):
        super().__init__()
        self.max_chars_per_document = max_chars_per_document
    
    def run(self, data, rank: int = 0, world_size: int = 1):
        from datatrove.data import Document
        for document in data:
            if len(document.text) <= self.max_chars_per_document:
                yield document
            else:
                # Split the document into smaller chunks
                chunks = self._split_text(document.text, self.max_chars_per_document)
                for i, chunk in enumerate(chunks):
                    # Create new document for each chunk
                    new_doc = Document(
                        text=chunk,
                        id=f"{document.id}_chunk_{i}" if document.id else f"chunk_{i}",
                        metadata=document.metadata.copy() if document.metadata else {}
                    )
                    yield new_doc
    
    def _split_text(self, text: str, max_chars: int) -> list[str]:
        """Split text into chunks, preferring to split on \\n\\n when possible."""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        remaining_text = text
        
        while len(remaining_text) > max_chars:
            # Find the best split point within the character limit
            split_point = self._find_best_split_point(remaining_text, max_chars)
            
            # Extract the chunk and update remaining text
            chunk = remaining_text[:split_point].rstrip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move to the next part, skipping any leading whitespace
            remaining_text = remaining_text[split_point:].lstrip()
        
        # Add the final chunk if it's not empty
        if remaining_text.strip():
            chunks.append(remaining_text.strip())
        
        return chunks
    
    def _find_best_split_point(self, text: str, max_chars: int) -> int:
        """Find the best point to split text within max_chars limit."""
        if len(text) <= max_chars:
            return len(text)
        
        # Try splitting on double newlines first
        double_newline_splits = []
        pos = 0
        while pos < max_chars:
            pos = text.find('\n\n', pos)
            if pos == -1 or pos >= max_chars:
                break
            double_newline_splits.append(pos + 2)  # Include the \n\n
            pos += 2
        
        if double_newline_splits:
            return double_newline_splits[-1]
        
        # Try splitting on single newlines
        single_newline_splits = []
        pos = 0
        while pos < max_chars:
            pos = text.find('\n', pos)
            if pos == -1 or pos >= max_chars:
                break
            single_newline_splits.append(pos + 1)  # Include the \n
            pos += 1
        
        if single_newline_splits:
            return single_newline_splits[-1]
        
        # Try splitting on sentence boundaries
        sentence_splits = []
        pos = 0
        while pos < max_chars:
            pos = text.find('. ', pos)
            if pos == -1 or pos >= max_chars:
                break
            sentence_splits.append(pos + 2)  # Include the '. '
            pos += 2
        
        if sentence_splits:
            return sentence_splits[-1]
        
        # Last resort: split at word boundaries
        last_space = text.rfind(' ', 0, max_chars)
        if last_space > 0:
            return last_space + 1
        
        # Absolute last resort: hard split at character limit
        return max_chars

class DoclingPostprocessing(PipelineStep):
    def run(self, data, rank: int = 0, world_size: int = 1):
        import re
        docling_tags_regex = r"(</docling_image>|<docling_image>|<docling_table>|</docling_table>|docling_picture_annotation(_non_text)?>|</docling_picture_annotation(_non_text)?>|<docling_formula>|</docling_formula>|<docling_page_number>|</docling_page_number>|<docling_boilerplate_top>|</docling_boilerplate_top>|<docling_boilerplate_bottom>|</docling_boilerplate_bottom>|<failed_page>|</failed_page>|<corrupted_page>|</corrupted_page>)"
        for document in data:
            # Remove all docling tags
            document.text = re.sub(docling_tags_regex, "", document.text)
            yield document

class Normalize(PipelineStep):
    def run(self, data, rank: int = 0, world_size: int = 1):
        import re
        for document in data:
            # Replace multiple newlines with single newlines
            document.text = re.sub(r'\n+', '\n', document.text)
            yield document


if __name__ == "__main__":
    args = parser.parse_args()
    # Output name should be the same as last part of the data path
    if args.name:
        output_name = args.name
    else:
        output_name = args.data_paths.replace("/", "_")
    print(f"Output name: {output_name}")

    data_paths = args.data_paths.split(",")
    print(f"Data paths: {data_paths}")

    from datatrove.executor import SlurmPipelineExecutor
    from datatrove.pipeline.filters import SamplerFilter
    from datatrove.pipeline.readers import JsonlReader, ParquetReader
    from datatrove.pipeline.writers import JsonlWriter
    from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer
    from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger

    # Select reader class based on --reader argument
    reader_class = JsonlReader if args.reader == "jsonl" else ParquetReader
    tokenizer_executor = SlurmPipelineExecutor(
        job_name=f"tok-{output_name}",
        pipeline=[
            reader_class(data_paths[0], text_key=args.text_key, shuffle_files=True, glob_pattern=args.glob_pattern),
            SamplerFilter(rate=args.sample),
            *([JsonlWriter(args.jsonl_output)] if args.jsonl_output else []),
            *([DoclingPostprocessing()] if args.enable_docling_postprocessing else []),
            *([DocumentSplitter(args.max_chars_per_document)] if args.max_chars_per_document else []),
            *([Normalize()] if args.enable_normalize else []),
            DocumentTokenizer(
                output_folder=f"s3://bucket/experiments/tokenized/{output_name}",
                local_working_dir=f"/scratch/user/tokenized/{output_name}",
                eos_token="<|end_of_text|>",
                tokenizer_name_or_path=args.tokenizer,
                batch_size=args.batch_size,
                max_tokens_per_file=args.max_toks,
                # Max 1 GT per file (i.e. btw 5 et 300 tokenized files per dump et about 100 dump extracts per merged file)
                shuffle_documents=True,
                shuffle_chunk_size=args.shuffle_chunk_size + 1 if args.shuffle_chunk_size else None
            ),
        ],
        tasks=args.n_tasks,
        time="20:00:00",
        partition="hopper-cpu",
        logging_dir=f"/shared/user/logs/pdf_project/experiments/tokenization/{output_name}/tokenized",
        cpus_per_task=8,
        mem_per_cpu_gb=2,
        qos=args.qos,
        env_command="sleep $((RANDOM % 30))",
        mail_user="user@example.com",
        depends_job_id=args.dep_job_id
    )
    # tokenizer_executor = LocalPipelineExecutor(
    #     pipeline=[
    #         *([JsonlReader(data_path, text_key=args.text_key, limit=args.limit) for data_path in data_paths]),
    #         DocumentBatchTester(target_length=100_000),
    #         DocumentTokenizer(
    #             output_folder=f"s3://bucket/experiments/tokenized/{output_name}",
    #             local_working_dir=f"/scratch/user/tokenized/{output_name}",
    #             eos_token="<|end_of_text|>",
    #             tokenizer_name_or_path=args.tokenizer,
    #             batch_size=args.batch_size,
    #         )
    #     ]
    # )

    if args.run_merger:
        merge_executor = SlurmPipelineExecutor(
                job_name=f"merge-{output_name}",
                pipeline=[
                DocumentTokenizerMerger(
                    input_folder=f"s3://bucket/experiments/tokenized/{output_name}",
                    output_folder=f"s3://bucket/experiments/tokenized_merged/{output_name}",
                    save_filename="tokenized_dataset",
                    shuffle_chunk_size=args.shuffle_chunk_size + 1 if args.shuffle_chunk_size else None
                ),
            ],
            tasks=1,
            time="20:00:00",
            partition="hopper-cpu",
            logging_dir=f"/shared/user/logs/pdf_project/experiments/{output_name}/tokenized_merged",
            cpus_per_task=10,
            mem_per_cpu_gb=2,
            qos=args.qos,
            depends=tokenizer_executor
        )

        merge_executor.run()
    else:
        tokenizer_executor.run() 