import streamlit as st
import sys
import io
from pathlib import Path
import pandas as pd

# --- Import your modules ---
# Assumes gui.py is in the same directory as the other scripts
sys.path.insert(0, str(Path(__file__).parent))
from label_process import stats, yolo_stats, convert_to_yolo
from split_dataset import merge_and_resplit_dataset, stratified_split_dataset

st.set_page_config(layout="wide")
st.title("YOLO Dataset Tools")

tab = st.sidebar.radio("Action", [
    "Main Page",
    "BSTLD Stats",
    "YOLO Stats",
    "Convert BSTLD → YOLO",
    "Re-split Dataset",
    "View Results"
])

# ─── Helper to capture stdout ───────────────────────────────────────────────
def capture(fn, *args, **kwargs):
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        fn(*args, **kwargs)
    finally:
        sys.stdout = old
    return buf.getvalue()

if tab == "Main Page":
    st.header("Main Page")
    with open("main_page_readme.md", "r", encoding="utf-8") as f:
        main_text = f.read()

    st.markdown(main_text)


# ─ BSTLD Stats ─────────────────────────────────────────────────────────────
if tab == "BSTLD Stats":
    st.header("BSTLD Stats")
    labels_file = st.text_input("Path to BSTLD labels YAML file")
    if st.button("Run") and labels_file:
        try:
            out = capture(stats, labels_file, "print")
            st.code(out)
        except Exception as e:
            st.error(str(e))

# ─── YOLO Stats ──────────────────────────────────────────────────────────────
elif tab == "YOLO Stats":
    st.header("YOLO Stats")
    dataset_dir = st.text_input("Path to YOLO dataset directory")
    splits = st.multiselect("Splits", ["train", "val", "test"], default=["train", "val"])
    if st.button("Run") and dataset_dir:
        try:
            out = capture(yolo_stats, dataset_dir, output="print", splits=tuple(splits))
            st.code(out)
        except Exception as e:
            st.error(str(e))

# ─── Convert ─────────────────────────────────────────────────────────────────
elif tab == "Convert BSTLD → YOLO":
    st.header("Convert BSTLD → YOLO")
    output_path  = st.text_input("Output path")
    train_labels = st.text_input("Train labels YAML (optional)")
    test_labels  = st.text_input("Test labels YAML (optional)")
    image_transfer = st.selectbox("Image transfer", ["hardlink", "symlink-rel", "symlink-abs", "copy"])
    merge_color  = st.checkbox("Merge color variants (e.g. RedLeft → Red)")
    start_id     = st.number_input("Start class ID", min_value=0, value=0, step=1)
    replace      = st.checkbox("Replace existing output without asking")

    if st.button("Convert") and output_path:
        try:
            out = capture(
                convert_to_yolo,
                output_path,
                train_labels=train_labels or None,
                test_labels=test_labels or None,
                image_transfer=image_transfer,
                merge_color_variants=merge_color,
                start_id=int(start_id),
                replace=replace,
            )
            st.code(out)
            st.success("Done!")
        except Exception as e:
            st.error(str(e))

# ─── Re-split ────────────────────────────────────────────────────────────────
elif tab == "Re-split Dataset":
    st.header("Re-split Dataset")
    input_dir   = st.text_input("Input dataset directory")
    output_dir  = st.text_input("Output dataset directory")
    mode        = st.radio("Split mode", ["balanced (stratified)", "random"])
    split_ratio = st.slider("Train ratio", 0.5, 0.95, 0.8, 0.05)
    seed        = st.number_input("Random seed", value=42, step=1)
    link_type   = st.selectbox("File transfer", ["hardlink", "symlink", "copy"])

    if st.button("Run") and input_dir and output_dir:
        try:
            fn = stratified_split_dataset if mode.startswith("balanced") else merge_and_resplit_dataset
            out = capture(fn,
                input_dir=input_dir,
                output_dir=output_dir,
                split_ratio=split_ratio,
                seed=int(seed),
                link_type=link_type,
            )
            st.code(out)
            st.success("Done!")
        except Exception as e:
            st.error(str(e))

# ─── View Results ────────────────────────────────────────────────────────────────
elif tab == "View Results":
    st.header("View Results")
 
    # ── Root directory input ──────────────────────────────────────────────────
    # myroot = "/home/akosti02/epl445/epl445-traffic-light-recognition/runs/detect/scratch_combined/"
    root_input = st.text_input("Root directory", value=str(Path.home()))
    root = Path(root_input)
 
    if not root.exists() or not root.is_dir():
        st.warning("Enter a valid directory path above.")
        st.stop()
 
    # ── Folder navigation via session state ───────────────────────────────────
    if "browser_path" not in st.session_state or st.sidebar.button("Reset to root"):
        st.session_state.browser_path = root
 
    # If the root input changed, reset
    if st.session_state.browser_path.parts[:len(root.parts)] != root.parts:
        st.session_state.browser_path = root
 
    current = st.session_state.browser_path
 
    # ── Breadcrumb ────────────────────────────────────────────────────────────
    parts = current.relative_to(root).parts
    crumb_cols = st.columns(len(parts) + 1)
    if crumb_cols[0].button(root.name or str(root)):
        st.session_state.browser_path = root
        st.rerun()
    for i, part in enumerate(parts):
        if crumb_cols[i + 1].button(part):
            st.session_state.browser_path = root.joinpath(*parts[: i + 1])
            st.rerun()
 
    st.caption(f"📂 `{current}`")
    st.divider()
 
    # ── List contents ─────────────────────────────────────────────────────────
    VIEWABLE = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".csv"}
 
    try:
        entries = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        st.error("Permission denied.")
        st.stop()
 
    dirs  = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file() and e.suffix.lower() in VIEWABLE]
    other = [e for e in entries if e.is_file() and e.suffix.lower() not in VIEWABLE]
 
    # Sub-folders
    if dirs:
        st.subheader("Folders")
        cols = st.columns(4)
        for i, d in enumerate(dirs):
            if cols[i % 4].button(f"📁 {d.name}", key=f"dir_{d}"):
                st.session_state.browser_path = d
                st.rerun()
 
    # Other files (non-viewable) — just list names
    if other:
        with st.expander(f"Other files ({len(other)})"):
            for f in other:
                st.text(f.name)
 
    # ── Viewable files ────────────────────────────────────────────────────────
    if not files:
        if not dirs:
            st.info("No PNG or CSV files found here.")
        st.stop()
 
    st.subheader(f"Files ({len(files)})")
 
    # Filter bar
    filter_ext = st.multiselect(
        "Show",
        options=["Images", "CSV"],
        default=["Images", "CSV"],
        label_visibility="collapsed",
    )
    show_exts = set()
    if "Images" in filter_ext:
        show_exts.update({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"})
    if "CSV" in filter_ext:
        show_exts.add(".csv")
 
    visible = [f for f in files if f.suffix.lower() in show_exts]
 
    if not visible:
        st.info("No files match the current filter.")
        st.stop()
 
    selected_name = st.selectbox("Select file", [f.name for f in visible])
    selected = current / selected_name
 
    st.divider()
 
    ext = selected.suffix.lower()
    if ext == ".csv":
        try:
            df = pd.read_csv(selected)
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    elif ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
        st.image(str(selected), caption=selected.name, use_container_width=True)



