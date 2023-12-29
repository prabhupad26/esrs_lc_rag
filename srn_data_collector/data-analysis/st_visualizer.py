import fitz  # PyMuPDF
import streamlit as st


def draw_bounding_boxes(pdf_path, bounding_boxes):
    doc = fitz.open(pdf_path)
    page = doc[0]  # Assume we are working with the first page

    for box in bounding_boxes:
        x0, y0, x1, y1 = box["x0"], box["y0"], box["x1"], box["y1"]
        rect = fitz.Rect(x0, y0, x1, y1)
        page.drawRect(rect, color=(1, 0, 0), width=2)

    return doc


def main():
    st.title("PDF Bounding Boxes Viewer")

    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])
    json_file = st.file_uploader("Upload JSON file with bounding boxes", type=["json"])

    if pdf_file and json_file:
        pdf_path = f"temp_{pdf_file.name}"
        with open(pdf_path, "wb") as pdf_temp:
            pdf_temp.write(pdf_file.read())

        bounding_boxes = json_file.read()
        bounding_boxes = st.json(bounding_boxes)

        st.write("### Bounding Boxes Coordinates:")
        st.write(bounding_boxes)

        st.write("### PDF with Bounding Boxes:")
        with st.spinner("Drawing bounding boxes..."):
            modified_pdf = draw_bounding_boxes(pdf_path, bounding_boxes)
            modified_pdf_path = f"modified_{pdf_file.name}"
            modified_pdf.save(modified_pdf_path)
            st.image(modified_pdf_path, use_container_width=True)

    st.warning("Please make sure the PDF file and JSON file are uploaded together.")


if __name__ == "__main__":
    main()
