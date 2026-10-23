"""
Builds the PUBLIC DEMO corpus — entirely synthetic, hand-authored sample
receipts, NOT derived from SROIE, WildReceipt, or CORD in any way.

Why synthetic: SROIE's official source (rrc.cvc.uab.es) gates the data
behind competition registration with no clear public-redistribution grant,
and WildReceipt's license is explicitly listed as "N/A" in the official
MMOCR docs. CORD alone is clearly CC-BY-4.0, but mixing one real licensed
source with two excluded ones adds confusion for no real benefit — the
demo exists to show the SYSTEM working, not to reproduce the research
corpus. The real 3,712-document research corpus stays local/private,
never served publicly.

Every record is clearly marked "source": "synthetic_demo" so it can never
be confused with research data in logs, responses, or evaluation reports.
Schema matches scripts/chunk_corpus.py's output exactly (chunk_id, doc_id,
source, text, ground_truth{vendor,date,amount,all_vendors,all_dates,all_amounts})
so the existing, unmodified indexing code (embed_index.py, BM25Retriever)
can build an index from this file with zero changes.

Usage:
    python demo_data/build_synthetic_demo_corpus.py
"""

import json
import os

# Hand-authored, obviously fictional receipts — fake vendor names, fake
# addresses, fake amounts. Covers a spread of question types (vendor lookup,
# date lookup, amount lookup, a "no vendor" edge case) so the demo UI's
# sample questions have real, correct answers to show.
SYNTHETIC_RECEIPTS = [
    {
        "vendor": "Example Grocers Inc.",
        "date": "03/14/2026",
        "amount": "42.17",
        "text": "EXAMPLE GROCERS INC.\n123 Sample Street, Faketown, ST 00000\n"
                 "Date: 03/14/2026\nBread 3.50\nMilk 4.20\nEggs 5.00\n"
                 "Apples (3 lb) 6.47\nCheese 8.00\nSubtotal 27.17\nTax 2.00\n"
                 "TOTAL 42.17\nThank you for shopping with us!",
    },
    {
        "vendor": "Sample Coffee Co.",
        "date": "01/02/2026",
        "amount": "9.75",
        "text": "SAMPLE COFFEE CO.\n456 Demo Avenue, Testville, ST 11111\n"
                 "Date: 01/02/2026\nLarge Latte 5.25\nBlueberry Muffin 4.50\n"
                 "TOTAL 9.75\nVisit us again soon!",
    },
    {
        "vendor": "Fictional Hardware Supply",
        "date": "11/20/2025",
        "amount": "156.42",
        "text": "FICTIONAL HARDWARE SUPPLY\n789 Placeholder Road, Mockcity, ST 22222\n"
                 "Date: 11/20/2025\nHammer 24.99\nNails (5lb box) 12.50\n"
                 "Paint (1 gal) 38.00\nPaintbrush set 15.93\nDrill bits 22.00\n"
                 "Extension cord 35.00\nSubtotal 148.42\nTax 8.00\nTOTAL 156.42",
    },
    {
        "vendor": "Demo Pizza Kitchen",
        "date": "03/14/2026",
        "amount": "28.90",
        "text": "DEMO PIZZA KITCHEN\n321 Imaginary Blvd, Sampleburg, ST 33333\n"
                 "Date: 03/14/2026\nLarge Pepperoni Pizza 18.99\nGarlic Bread 5.99\n"
                 "Soda (2L) 3.92\nSubtotal 28.90\nTOTAL 28.90\nEnjoy your meal!",
    },
    {
        "vendor": "Placeholder Pharmacy",
        "date": "09/05/2025",
        "amount": "63.18",
        "text": "PLACEHOLDER PHARMACY\n55 Testing Lane, Exampleville, ST 44444\n"
                 "Date: 09/05/2025\nPrescription copay 25.00\nVitamins 14.99\n"
                 "Hand sanitizer 4.19\nBandages 6.50\nAllergy medicine 12.50\n"
                 "TOTAL 63.18",
    },
    {
        "vendor": "Sample Gas Station",
        "date": "02/18/2026",
        "amount": "54.00",
        "text": "SAMPLE GAS STATION\nPump No. 4\n123 Fuel Way, Mockton, ST 55555\n"
                 "Date: 02/18/2026\nUnleaded Regular 14.2 GAL @ 3.80/GAL\n"
                 "TOTAL 54.00\nThank you, drive safely!",
    },
    {
        "vendor": "Mock Bookstore",
        "date": "12/24/2025",
        "amount": "37.45",
        "text": "MOCK BOOKSTORE\n88 Fiction Street, Demotown, ST 66666\n"
                 "Date: 12/24/2025\n'The Sample Chronicles' 15.99\n"
                 "'Placeholder Poems' 12.50\nBookmark 2.00\nGift wrap 3.00\n"
                 "Subtotal 33.49\nTax 3.96\nTOTAL 37.45",
    },
    {
        "vendor": "Example Electronics",
        "date": "04/01/2026",
        "amount": "299.99",
        "text": "EXAMPLE ELECTRONICS\n200 Circuit Ave, Testopolis, ST 77777\n"
                 "Date: 04/01/2026\nWireless Headphones 299.99\nTOTAL 299.99\n"
                 "Warranty: 1 year, see sample-electronics.example for terms.",
    },
    {
        # No vendor field — mirrors CORD-style structured receipts that
        # lack a vendor line in this project's real corpus (Section 3/13).
        "vendor": None,
        "date": None,
        "amount": "18.50",
        "text": "Menu\nSample Soup 6.00\nDemo Salad 8.50\nSparkling Water 4.00\n"
                 "Total 18.50",
    },
    {
        "vendor": "Demo Clothing Outlet",
        "date": "07/11/2025",
        "amount": "84.33",
        "text": "DEMO CLOTHING OUTLET\n410 Fashion Row, Sampleford, ST 88888\n"
                 "Date: 07/11/2025\nT-Shirt (x2) 29.98\nJeans 45.00\nSocks 5.99\n"
                 "Subtotal 80.97\nTax 3.36\nTOTAL 84.33",
    },
]


def build_record(idx, receipt):
    vendor = receipt["vendor"]
    date = receipt["date"]
    amount = receipt["amount"]
    return {
        "chunk_id": f"demo_{idx:03d}",
        "doc_id": f"synthetic_demo_{idx:03d}",
        "source": "synthetic_demo",
        "text": receipt["text"],
        "char_len": len(receipt["text"]),
        "word_len": len(receipt["text"].split()),
        "ground_truth": {
            "vendor": vendor,
            "date": date,
            "amount": amount,
            "all_vendors": [vendor] if vendor else [],
            "all_dates": [date] if date else [],
            "all_amounts": [amount] if amount else [],
        },
    }


def main():
    out_path = os.path.join(os.path.dirname(__file__), "demo_chunks.jsonl")
    records = [build_record(i, r) for i, r in enumerate(SYNTHETIC_RECEIPTS)]
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} synthetic demo receipts -> {out_path}")
    print("All records tagged source='synthetic_demo' — not derived from "
          "SROIE/WildReceipt/CORD, safe for public deployment.")


if __name__ == "__main__":
    main()
