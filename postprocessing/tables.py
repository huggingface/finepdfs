import re
import logging
from typing import List, Optional
from datatrove.data import DocumentsPipeline
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.filters.language_filter import LanguageFilter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.utils.media import iter_pages
import numpy as np
import argparse
from loguru import logger


class TableParser:
    """Helper class to parse markdown tables within docling_table tags"""
    
    @staticmethod
    def parse_table(table_content: str) -> Optional[List[List[str]]]:
        """Parse markdown table content into a 2D list of cells"""
        lines = table_content.strip().split('\n')
        if len(lines) < 2:
            return None
            
        parsed_rows = []
        for i, line in enumerate(lines):
            if i == 1 and '---' in line:  # Skip separator row
                continue
            if line.strip().startswith('|') and line.strip().endswith('|'):
                # Parse table row, removing leading/trailing pipes and splitting
                cells = [cell.strip() for cell in line.strip()[1:-1].split('|')]
                parsed_rows.append(cells)
        
        return parsed_rows if parsed_rows else None
    
    @staticmethod
    def table_to_markdown(rows: List[List[str]]) -> str:
        """Convert parsed table back to markdown format"""
        if not rows:
            return ""
        
        result = []
        for i, row in enumerate(rows):
            row_str = '| ' + ' | '.join(row) + ' |'
            result.append(row_str)
            
            # Add separator after header row
            if i == 0:
                separator = '|' + '|'.join(['---' for _ in row]) + '|'
                result.append(separator)
        
        return '\n'.join(result)
    
    @staticmethod
    def has_empty_headers(rows: List[List[str]]) -> bool:
        """Check if all headers are empty"""
        if not rows:
            return True
        header_row = rows[0]
        return all(cell.strip() == '' for cell in header_row)
    
    @staticmethod
    def has_empty_column(rows: List[List[str]]) -> bool:
        """Check if any column is completely empty"""
        if not rows:
            return True
        
        num_cols = len(rows[0]) if rows else 0
        for col_idx in range(num_cols):
            if all(row[col_idx].strip() == '' for row in rows if col_idx < len(row)):
                return True
        return False
    
    @staticmethod
    def has_empty_row(rows: List[List[str]]) -> bool:
        """Check if any row is completely empty"""
        for row in rows:
            if all(cell.strip() == '' for cell in row):
                return True
        return False
    
    @staticmethod
    def clean_table(rows: List[List[str]]) -> List[List[str]]:
        """Remove empty columns and rows from table"""
        if not rows:
            return rows
        
        # Find columns to keep (not empty and header not empty)
        num_cols = max(len(row) for row in rows) if rows else 0
        cols_to_keep = []
        
        for col_idx in range(num_cols):
            # Check if column has content or header is not empty
            has_content = False
            header_not_empty = False
            
            for row_idx, row in enumerate(rows):
                if col_idx < len(row):
                    if row_idx == 0 and row[col_idx].strip() != '':  # Header not empty
                        header_not_empty = True
                    if row[col_idx].strip() != '':  # Cell has content
                        has_content = True
            
            # Keep column if it has either non-empty header OR content
            if header_not_empty or has_content:
                cols_to_keep.append(col_idx)
        
        # Filter columns
        filtered_rows = []
        for row in rows:
            filtered_row = [row[col_idx] if col_idx < len(row) else '' 
                        for col_idx in cols_to_keep]
            filtered_rows.append(filtered_row)
        
        # Remove completely empty rows
        cleaned_rows = []
        for row in filtered_rows:
            if any(cell.strip() != '' for cell in row):
                cleaned_rows.append(row)
        
        return cleaned_rows

class RemoveAllTables(PipelineStep):
    """Pipeline step that removes all docling_table content completely"""
    
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        import re
        table_regex = re.compile(r"<docling_table>.*?</docling_table>", re.DOTALL)
        
        for document in data:
            # Remove all table content
            document.text = table_regex.sub("", document.text)
            yield document


class RemoveProblematicTables(PipelineStep):
    """Pipeline step that removes tables with empty headers, columns, or rows"""
    
    def __init__(self):
        self.processed_tables = 0
        self.removed_tables = 0
        self.removal_reasons = {"malformed": 0, "empty_headers": 0, "empty_columns": 0, "empty_rows": 0}
    
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        import re
        table_regex = re.compile(r"<docling_table>(.*?)</docling_table>", re.DOTALL)
        
        for document in data:
            def replace_table(match):
                self.processed_tables += 1
                table_content = match.group(1)
                rows = TableParser.parse_table(table_content)
                
                if not rows:
                    self.removed_tables += 1
                    self.removal_reasons["malformed"] += 1
                    logger.debug(f"Removed malformed table (couldn't parse): {table_content[:100]}...")
                    return ""  # Remove malformed tables
                
                # Check for problematic conditions
                removal_reasons = []
                if TableParser.has_empty_headers(rows):
                    removal_reasons.append("empty_headers")
                    self.removal_reasons["empty_headers"] += 1
                if TableParser.has_empty_column(rows):
                    removal_reasons.append("empty_columns") 
                    self.removal_reasons["empty_columns"] += 1
                if TableParser.has_empty_row(rows):
                    removal_reasons.append("empty_rows")
                    self.removal_reasons["empty_rows"] += 1
                
                if removal_reasons:
                    self.removed_tables += 1
                    logger.debug(f"Removed table due to: {', '.join(removal_reasons)}")
                    logger.debug(f"Table content preview: {TableParser.table_to_markdown(rows)[:200]}...")
                    return ""  # Remove problematic tables
                
                logger.debug(f"Kept table with {len(rows)} rows and {len(rows[0]) if rows else 0} columns")
                return match.group(0)  # Keep good tables
            
            document.text = table_regex.sub(replace_table, document.text)
            yield document
        
        # Log summary statistics
        logger.info(f"RemoveProblematicTables Summary:")
        logger.info(f"  Processed tables: {self.processed_tables}")
        logger.info(f"  Removed tables: {self.removed_tables}")
        logger.info(f"  Kept tables: {self.processed_tables - self.removed_tables}")
        logger.info(f"  Removal reasons: {self.removal_reasons}")


class CleanTables(PipelineStep):
    """Pipeline step that cleans tables by removing empty columns and rows"""
    name: str = "🧹 CleanTables"
    type: str = "🔧 Formatter"

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        import re
        table_regex = re.compile(r"<docling_table>(.*?)</docling_table>", re.DOTALL)
        
        for document in data:
            def replace_table(match):
                self.stat_update("processed_tables", 1, unit="document")
                table_content = match.group(1)
                rows = TableParser.parse_table(table_content)
                
                if not rows:
                    self.stat_update("removed_tables", 1, unit="document")
                    return ""  # Remove malformed tables
                
                original_rows = len(rows)
                original_cols = max(len(row) for row in rows) if rows else 0
                
                # Clean the table
                cleaned_rows = TableParser.clean_table(rows)
                
                if not cleaned_rows:
                    self.stat_update("removed_tables", 1, unit="document")
                    return ""  # Remove if nothing left after cleaning
                
                cleaned_table_rows = len(cleaned_rows)
                cleaned_table_cols = len(cleaned_rows[0]) if cleaned_rows else 0
                
                rows_removed = original_rows - cleaned_table_rows
                cols_removed = original_cols - cleaned_table_cols
                
                self.stat_update("rows_removed_count", rows_removed, unit="document")
                self.stat_update("columns_removed_count", cols_removed, unit="document")
                
                if rows_removed > 0 or cols_removed > 0:
                    self.stat_update("cleaned_tables", 1, unit="document")
                
                # Convert back to markdown and wrap in docling tags
                cleaned_markdown = TableParser.table_to_markdown(cleaned_rows)
                return f"<docling_table>{cleaned_markdown}</docling_table>"
            
            with self.track_time():
                pages = []
                for page in iter_pages(document):
                    pages.append(table_regex.sub(replace_table, page))
                document.text = "".join(pages)
                document.media[0].metadata["page_offsets"] = [int(offset) for offset in np.cumsum([len(page) for page in pages])]
            yield document