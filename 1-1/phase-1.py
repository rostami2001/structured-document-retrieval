import fitz
import json
import re

def extract_text_elements_from_pdf(pdf_path):
    elements = []
    with fitz.open(pdf_path) as doc:
        for page_num in range(doc.page_count):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] == 0:
                    for line in block["lines"]:
                        line_text = ''.join([span.get("text", "") for span in line["spans"]]).strip()
                        font_size = max([span.get("size", 0) for span in line["spans"]])
                        elements.append({
                            "text": line_text,
                            "font_size": font_size
                        })
    return elements

def parse_text_elements(elements):
    data = {}
    current_main_title = None
    current_subtitle = None
    title_buffer = ""
    subtitle_buffer = ""
    content_buffer = []
    list_buffer = []
    in_list = False

    avg_font_size = sum(el["font_size"] for el in elements) / len(elements)

    for i, el in enumerate(elements):
        line = el["text"]
        font_size = el["font_size"]

        # main titles
        if font_size > avg_font_size * 1.5:
            # Save any content for previous main title with no subtitles
            if current_main_title and not current_subtitle and content_buffer:
                content_text = " ".join(content_buffer).strip()
                short_content = " ".join(content_text.split()[:10]) + " ..."
                data[current_main_title] = short_content
                content_buffer = []

            if title_buffer:
                title_buffer += " " + line
            else:
                title_buffer = line

            if i + 1 < len(elements) and elements[i + 1]["font_size"] > avg_font_size * 1.5:
                continue

            # Finalize main title
            current_main_title = title_buffer.replace(" ", "")
            data[current_main_title] = {}
            current_subtitle = None
            title_buffer = ""

        # subtitles
        elif avg_font_size < font_size <= avg_font_size * 1.5 and not re.search(r'\d', line):
            # Save any content for previous main title with no subtitles
            if current_main_title and not current_subtitle and content_buffer:
                content_text = " ".join(content_buffer).strip()
                short_content = " ".join(content_text.split()[:10]) + " ..."
                data[current_main_title] = short_content
                content_buffer = []

            if subtitle_buffer:
                subtitle_buffer += " " + line
            else:
                subtitle_buffer = line

            if i + 1 < len(elements) and avg_font_size < elements[i + 1]["font_size"] <= avg_font_size * 1.5 and not re.search(r'\d', elements[i + 1]["text"]):
                continue

            # Finalize subtitle
            current_subtitle = subtitle_buffer.replace(" ", "")
            data[current_main_title][current_subtitle] = []
            subtitle_buffer = ""
            list_buffer = []
            in_list = False

        # list items
        elif re.match(r"^\d+\.\s", line) or re.match(r"^[A-Z]\.\s", line):
            # Only process list items if current title and subtitle exist
            if current_main_title and current_subtitle:
                # If we're already in a list, finalize the current list item
                if in_list and list_buffer:
                    list_text = " ".join(list_buffer).strip()
                    list_content = " ".join(list_text.split()[:10]) + " ..."
                    data[current_main_title][current_subtitle].append(list_content)
                    list_buffer = []

                list_buffer.append(line)
                in_list = True

        # Handle multi-line list items
        elif in_list and current_main_title and current_subtitle:
            list_buffer.append(line)

            # Finalize the list item if the next line is a new title or subtitle
            if i + 1 < len(elements) and (
                elements[i + 1]["font_size"] > avg_font_size * 1.5 or
                (avg_font_size < elements[i + 1]["font_size"] <= avg_font_size * 1.5 and not re.search(r'\d', elements[i + 1]["text"]))
            ):
                list_text = " ".join(list_buffer).strip()
                list_content = " ".join(list_text.split()[:10]) + " ..."
                data[current_main_title][current_subtitle].append(list_content)
                list_buffer = []
                in_list = False

        # regular content
        elif current_main_title:
            content_buffer.append(line)

            # Finalize content when reaching the next title or subtitle
            if i + 1 < len(elements) and (
                elements[i + 1]["font_size"] > avg_font_size * 1.5 or
                (avg_font_size < elements[i + 1]["font_size"] <= avg_font_size * 1.5 and not re.search(r'\d', elements[i + 1]["text"]))
            ):
                content_text = " ".join(content_buffer).strip()
                short_content = " ".join(content_text.split()[:10]) + " ..."
                if current_subtitle:
                    data[current_main_title][current_subtitle] = short_content
                else:
                    data[current_main_title] = short_content
                content_buffer = []
                in_list = False

    # Save the last buffer if any content remains
    if current_main_title:
        if in_list and list_buffer:
            data[current_main_title][current_subtitle].append(" ".join(list_buffer).strip())
        elif content_buffer:
            content_text = " ".join(content_buffer).strip()
            short_content = " ".join(content_text.split()[:10]) + " ..."
            if current_subtitle:
                data[current_main_title][current_subtitle] = short_content
            else:
                data[current_main_title] = short_content

    return data

def save_to_json(data, output_path="output.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

pdf_path = 'E:\\university\\semester-9\\information retrieval\\project\\1-1\\Depressive Disorders.pdf'

elements = extract_text_elements_from_pdf(pdf_path)
parsed_data = parse_text_elements(elements)
save_to_json(parsed_data)
