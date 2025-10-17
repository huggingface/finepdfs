import fitz
import pymupdf
from typing import Optional
from docling_core.types.doc.base import CoordOrigin, BoundingBox

from docling_core.types.doc.document import (
    TableItem,
    TableData,
    TableCell,
)
import logging

_log = logging.getLogger(__name__)

def update_pymupdf_table(pymupdf_page: pymupdf.Page, table_item: TableItem) -> Optional[TableItem]:
        # Convert to top-left origin if needed
        if table_item.prov[0].bbox.coord_origin != CoordOrigin.TOPLEFT:
            page_height = pymupdf_page.cropbox.height
            target_bbox = table_item.prov[0].bbox.to_top_left_origin(page_height)
        else:
            target_bbox = table_item.prov[0].bbox
        
        # Extract table using PyMuPDF with the bounding box as clip
        extracted_table_data = _extract_table_from_bbox(
            pymupdf_page, target_bbox
        )
        
        if extracted_table_data:
            # Update the table item with the extracted data
            table_item.data = extracted_table_data
            _log.info(f"Updated table with {extracted_table_data.num_rows} rows and {extracted_table_data.num_cols} columns")

        return table_item

def _extract_table_from_bbox(pymupdf_page: pymupdf.Page, target_bbox: BoundingBox) -> Optional[TableData]:
    """
    Extract table data from a specific bounding box using PyMuPDF.
    
    Args:
        pymupdf_page: The PyMuPDF page object
        target_bbox: The bounding box to extract from
        table_item: The original table item for reference
        
    Returns:
        TableData object with extracted content or None if extraction fails
    """
    try:
        # Convert bounding box to PyMuPDF Rect
        rect = fitz.Rect(target_bbox.l, target_bbox.t, target_bbox.r, target_bbox.b)
        
        # Find tables in the specified area
        tabs = pymupdf_page.find_tables(clip=rect, strategy="lines")
        
        if not tabs.tables:
            _log.warning(f"No tables found in the specified bbox {rect}")
            return None
        
        # Use the first (and likely only) table found in the clipped area
        if len(tabs.tables) > 1:
            _log.warning(f"Multiple tables found in the specified bbox {rect}")
            return None
        
        pymupdf_table = tabs.tables[0]
        
        # Extract table structure and content
        table_content = pymupdf_table.extract()
        
        if not table_content:
            return None
        
        num_rows = len(table_content)
        num_cols = len(table_content[0]) if table_content else 0
        
        if num_rows == 0 or num_cols == 0:
            return None
        
        # Create TableData structure
        table_cells = []
        
        for row_idx, row in enumerate(table_content):
            for col_idx, cell_text in enumerate(row):
                if cell_text is None:
                    cell_text = ""
                
                # Get cell bbox if available
                cell_bbox = None
                if hasattr(pymupdf_table, 'cells') and pymupdf_table.cells:
                    try:
                        # Calculate cell index in the flattened cells list
                        cell_index = row_idx * num_cols + col_idx
                        if cell_index < len(pymupdf_table.cells):
                            cell_rect = pymupdf_table.cells[cell_index]
                            if cell_rect:
                                # Convert PyMuPDF rect to BoundingBox
                                cell_bbox = BoundingBox(
                                    l=cell_rect[0],
                                    t=cell_rect[1], 
                                    r=cell_rect[2],
                                    b=cell_rect[3],
                                    coord_origin=CoordOrigin.TOPLEFT
                                )
                    except (IndexError, TypeError):
                        # Cell bbox not available, continue without it
                        pass
                
                # Determine if this is a header cell
                is_column_header = False
                if hasattr(pymupdf_table, 'header') and pymupdf_table.header:
                    # Check if this row is part of the header
                    if not pymupdf_table.header.external:
                        # Header is internal (first row(s) of the table)
                        is_column_header = (row_idx == 0)
                    # For external headers, we'll need more complex logic
                    # For now, assume first row is header if we can't determine otherwise
                    else:
                        is_column_header = (row_idx == 0)
                else:
                    # Default assumption: first row is header
                    is_column_header = (row_idx == 0)

                cell_text_normalized = str(cell_text).encode("utf-8", errors="replace").decode("utf-8", errors="replace")
                
                cell = TableCell(
                    text=cell_text_normalized,
                    row_span=1,  # PyMuPDF doesn't provide span info directly
                    col_span=1,
                    start_row_offset_idx=row_idx,
                    end_row_offset_idx=row_idx + 1,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + 1,
                    column_header=is_column_header,
                    row_header=False,  # Could be enhanced to detect row headers
                    bbox=cell_bbox
                )
                table_cells.append(cell)
        
        table_data = TableData(
            num_rows=num_rows,
            num_cols=num_cols,
            table_cells=table_cells
        )
        
        return table_data
        
    except Exception as e:
        _log.error(f"Error extracting table from bbox {target_bbox}: {e}")