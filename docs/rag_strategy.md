1. Multimodal Parsing:
    - Text and tables: Use LLamaParse eller gpt-4o to extract financial tables to a Markdown typed file to preserve column structures before chunking.

    - Org-schemes (images): convert images to a JSON description through the gpt-4o Vision API. Vectorize the JSON text.

2. Chunking structure:
    - Double linked list: create a double linked structure through metadata (chunk_id, previous_chunk_id, next_chunk_id) to let the agent retrieve the previous or subsequent context on accord.

3. Valid enums (validation):
    - doc_type: (agreement, financial_report, minutes, org_chart, other)
    - category: (HR, Financing, Legal, Operations, other)

4. Qdrant Payload Schema (Metadata):
    - every vector point must contain the follow structure in its payload to enable the necessary filtering:

    {
        "raw_text": "...the actual text chunk...", // Necessary to generate the actual answer
        "metadata": {
            "tenant_id": "default_tenant_test_corp_123", // String
            "company_name": "SME Corp AB", // String
            "doc_type": "financial_report", // Enum
            "category": "Financing", // Enum
            "year": 2024, // Integer
            "quarter": 3, // Integer (1-4) or null
            "source_filename": "q3_report_2024.pdf", // String
            "page_number": 14, // Integer
            "chunk_id": "uuid123", // String (Unique)
            "previous_chunk_id": "uuid122", // String or null
            "next_chunk_id": "uuid124", // String or null
            "language": "sv", // String
            "processed_timestamp": "2026-08-09T14:30:00Z" // DateTime (ISO 8601)
        }
    }