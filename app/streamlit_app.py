"""AutoNet Vision Studio: photo labels, person detection, and CNN experiments."""

import hashlib
import html
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import (
    CLASSES, PHOTO_WEIGHTS, file_sha256, load_cifar_classifier,
    load_photo_classifier, prepare_image,
)
from src.people import PEOPLE_WEIGHTS, annotate_people, load_person_detector
from src.vision import scan_image

torch.set_num_threads(min(4, os.cpu_count() or 1))
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
CHECKPOINTS = {
    "Original NAS": MODELS / "best_nas_model.pth",
    "Improved NAS": MODELS / "improved_nas_model.pth",
    "Advanced CNN": MODELS / "advanced_cnn.pth",
    "Continued CNN": MODELS / "continued_cnn.pth",
}
MODES = {"Smart scan": "smart", "Objects": "photo", "People": "people", "CIFAR-10": "cifar"}
PAGES = ("Image Studio", "Model Results", "Architecture Lab", "Training History")

st.set_page_config(
    page_title="AutoNet | Vision Studio", page_icon="◉", layout="wide",
)
st.markdown("<style>" + (ROOT / "app" / "studio.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)


def read_json(name, default=None):
    path = RESULTS / name
    if not path.is_file():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st.warning(f"Could not read the saved result: {name}")
        return {} if default is None else default


def stamp(path):
    return path.stat().st_mtime_ns if path.is_file() else 0


@st.cache_resource(show_spinner=False)
def get_classifier(kind, path, modified_ns):
    return load_photo_classifier(path) if kind == "photo" else load_cifar_classifier(path)


@st.cache_resource(show_spinner=False)
def get_detector(path, modified_ns):
    return load_person_detector(path)


def accuracies():
    comparison = read_json("model_comparison.json")
    values = {name: float(comparison[name]) for name in ("Original NAS", "Improved NAS") if name in comparison}
    advanced = read_json("advanced_evaluation.json")
    if "test_accuracy" in advanced:
        values["Advanced CNN"] = float(advanced["test_accuracy"])
    continued = read_json("continued_evaluation.json")
    if CHECKPOINTS["Continued CNN"].is_file() and continued.get("checkpoint_sha256") == file_sha256(CHECKPOINTS["Continued CNN"]):
        values["Continued CNN"] = float(continued["test_accuracy"])
    verified = read_json("verified_evaluation.json")
    for name, path in CHECKPOINTS.items():
        check = verified.get("models", {}).get(path.name, {})
        if path.is_file() and check.get("checkpoint_sha256") == file_sha256(path):
            values[name] = float(check["test_accuracy"])
    return values


def masthead():
    with st.container(key="studio_masthead"):
        brand, nav = st.columns([1, 2.5], gap="medium", vertical_alignment="center")
        brand.markdown(
            '<div class="studio-brand">'
            '<svg class="studio-mark" viewBox="0 0 40 40" fill="none" aria-hidden="true">'
            '<rect width="40" height="40" rx="8" fill="#171717"/>'
            '<path d="M12 17v-5h5m6 0h5v5m0 6v5h-5m-6 0h-5v-5" stroke="white" stroke-width="1.5"/>'
            '<circle cx="20" cy="20" r="3" fill="white"/></svg>'
            '<div><strong>AutoNet</strong><span>VISION STUDIO</span></div></div>',
            unsafe_allow_html=True,
        )
        return nav.radio(
            "Workspace", PAGES, horizontal=True, key="studio_page",
            label_visibility="collapsed", width="stretch",
        )


def page_heading(eyebrow, title, subtitle):
    st.markdown(
        f'<div class="page-heading"><div class="section-label">{html.escape(eyebrow)}</div>'
        f'<h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_canvas(preview, actual, people_detected):
    with st.container(key="image_canvas"):
        dimensions = f"{preview.width:,} × {preview.height:,} px" if preview is not None else "Awaiting image"
        st.markdown(
            '<div class="canvas-bar"><span>IMAGE PREVIEW</span>'
            f'<span>{dimensions}</span></div>',
            unsafe_allow_html=True,
        )
        if preview is None:
            st.markdown(
                '<div class="canvas-empty"><svg viewBox="0 0 72 56" fill="none" aria-hidden="true">'
                '<path d="M2 16V2h14m40 0h14v14m0 24v14H56m-40 0H2V40"'
                ' stroke="currentColor" stroke-width="1.5"/>'
                '<path d="M30 28h12m-6-6v12" stroke="currentColor" stroke-width="1.5"/>'
                '</svg><strong>No image loaded</strong><p>Upload a photo using the controls above.</p></div>',
                unsafe_allow_html=True,
            )
            return
        st.image(preview, width="stretch")
        status = "Detected people" if people_detected else "Original image"
        if actual:
            status += " · Dataset label " + actual
        st.markdown(f'<div class="canvas-status">{html.escape(status)}</div>', unsafe_allow_html=True)


def render_result(result, preview, actual=None):
    if result["kind"] == "people":
        people = result["people"]
        title = "1 person" if len(people) == 1 else f"{len(people)} people"
        st.markdown(
            '<div class="result-summary"><div class="section-label">PERSON DETECTION</div>'
            f'<h3>{title}</h3><p>Highest detection score {people[0]["score_percent"]:.1f}%</p></div>',
            unsafe_allow_html=True,
        )
        rows = [{"Detection": f"Person {i}", "Score (%)": item["score_percent"]} for i, item in enumerate(people, 1)]
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, width="stretch",
            column_config={"Score (%)": st.column_config.ProgressColumn(
                "Detection score", min_value=0, max_value=100, format="%.1f%%",
            )},
        )
        buffer = io.BytesIO()
        preview.save(buffer, format="PNG")
        st.download_button("Download marked image", buffer.getvalue(), file_name="autonet_people.png", mime="image/png", width="stretch")
    elif result["kind"] == "no_people":
        st.markdown('<div class="result-summary"><h3>No people detected</h3></div>', unsafe_allow_html=True)
        st.caption("Try a clearer upper-body or full-body photo, or adjust the minimum score in Detection settings.")
    else:
        ranked = result["ranked"]
        title = html.escape(ranked[0]["class"].replace("_", " ").title())
        st.markdown(
            '<div class="result-summary"><div class="section-label">IMAGE CLASSIFICATION</div>'
            f'<h3>{title}</h3><p>Top-label score {ranked[0]["score_percent"]:.2f}%</p></div>',
            unsafe_allow_html=True,
        )
        rows = pd.DataFrame(ranked).rename(columns={"class": "Label", "score_percent": "Score (%)"})
        st.dataframe(
            rows, hide_index=True, width="stretch",
            column_config={"Score (%)": st.column_config.ProgressColumn(
                "Model score", min_value=0, max_value=100, format="%.2f%%",
            )},
        )
        if actual:
            if ranked[0]["class"] == actual:
                st.success(f"Correct: the dataset label is {actual}.")
            else:
                st.warning(f"The dataset label is {actual}; the model predicted {ranked[0]['class']}.")
    with st.expander("Model details"):
        st.caption(f"Model: {result['model']}")
        if result.get("dataset"):
            st.caption("Label set: " + result["dataset"])
        if result.get("people_checked"):
            st.caption("Smart scan checked for people before classifying the main object.")


def image_studio():
    page_heading(
        "01 / IMAGE STUDIO", "Scan a photo",
        "Detect people, classify an everyday object, or test a trained CIFAR-10 model.",
    )
    descriptions = {
        "smart": "Looks for people first. When none are detected, identifies the main object.",
        "photo": "Classifies the main object using 1,000 ImageNet labels.",
        "people": "Finds and counts people above the selected detection score.",
        "cifar": "Trained classes are " + ", ".join(CLASSES) + ".",
    }
    image_bytes = None
    image = None
    actual = None
    selected = None
    path = None
    with st.container(key="scan_toolbar", border=True):
        mode_col, file_col, action_col = st.columns([1.33, 1.32, .57], gap="medium", vertical_alignment="bottom")
        with mode_col:
            mode_label = st.radio("Scan mode", list(MODES), horizontal=True, key="scan_mode")
            mode = MODES[mode_label]
            if mode == "cifar":
                available = [name for name, candidate in CHECKPOINTS.items() if candidate.is_file()]
                if available:
                    preferred = "Continued CNN" if "Continued CNN" in available else available[-1]
                    selected = st.selectbox("Trained model", available, index=available.index(preferred), key="cifar_model")
                    path = CHECKPOINTS[selected]
                else:
                    st.error("No trained CIFAR-10 models were found.")
        with file_col:
            use_upload = mode != "cifar"
            if mode == "cifar" and path is not None:
                input_source = st.radio(
                    "Image source", ["Upload a photo", "Labelled test example"],
                    horizontal=True, key="cifar_source",
                )
                use_upload = input_source == "Upload a photo"
                if not use_upload:
                    examples = sorted((RESULTS / "prediction_examples").glob("test_*.png"))
                    if examples:
                        chosen = st.selectbox(
                            "Test image", examples, format_func=lambda value: value.stem,
                            key="cifar_example",
                        )
                        image_bytes = chosen.read_bytes()
                        actual = chosen.stem.split("_", 2)[-1]
                    else:
                        st.info("Generate test images with python -m src.verify_predictions.")
            if use_upload:
                upload = st.file_uploader("Image file", type=["jpg", "jpeg", "png", "webp"], key="studio_upload")
                if upload is not None:
                    image_bytes = upload.getvalue()
            if image_bytes is not None:
                try:
                    with Image.open(io.BytesIO(image_bytes)) as source:
                        image = prepare_image(source)
                except Exception as exc:
                    st.error(f"Could not read this image: {exc}")
        run = action_col.button("Scan image", type="primary", width="stretch", disabled=image is None, key="scan_image")
        st.caption(descriptions[mode])

    canvas_col, results_col = st.columns([1.85, 1], gap="medium")
    with results_col, st.container(key="scan_inspector", border=True):
        st.subheader("Scan result")
        output = st.container()
        threshold = 0.55
        if mode in {"smart", "people"}:
            with st.expander("Detection settings"):
                threshold = st.slider(
                    "Minimum person score", 0.30, 0.90, 0.55, 0.05, key="person_threshold",
                    help="A lower value shows weaker person detections.",
                )
                st.caption("People detector " + ("ready locally" if PEOPLE_WEIGHTS.is_file() else "downloads on first use"))
                if mode == "smart":
                    st.caption("Object model " + ("ready locally" if PHOTO_WEIGHTS.is_file() else "downloads on first use"))

    files = {
        "smart": (PEOPLE_WEIGHTS, PHOTO_WEIGHTS),
        "people": (PEOPLE_WEIGHTS,),
        "photo": (PHOTO_WEIGHTS,),
        "cifar": (path,) if path else (),
    }[mode]
    input_key = (
        mode, selected, threshold if mode in {"smart", "people"} else None,
        tuple((str(candidate), stamp(candidate)) for candidate in files),
        hashlib.sha256(image_bytes).hexdigest() if image_bytes is not None else None,
    )
    with output:
        if run:
            st.session_state.pop("studio_result", None)
            try:
                with st.spinner("Scanning your image..."):
                    detector = get_detector(str(PEOPLE_WEIGHTS), stamp(PEOPLE_WEIGHTS)) if mode in {"smart", "people"} else None
                    classifier = get_classifier("cifar", str(path), stamp(path)) if mode == "cifar" else None
                    result = scan_image(
                        image, mode, threshold, classifier, detector,
                        photo_classifier_loader=lambda: get_classifier("photo", str(PHOTO_WEIGHTS), stamp(PHOTO_WEIGHTS)),
                    )
                st.session_state["studio_result"] = {"input_key": input_key, "result": result}
            except Exception as exc:
                st.error(f"The scan could not finish: {exc}")
        record = st.session_state.get("studio_result", {})
        result = record.get("result") if record.get("input_key") == input_key and image is not None else None
        people_detected = result is not None and result["kind"] == "people"
        preview = annotate_people(image, result["people"]) if people_detected else image
        if result:
            render_result(result, preview, actual)
        else:
            st.markdown(
                '<div class="inspector-empty"><strong>No scan yet</strong>'
                '<p>Upload an image and select Scan image to see the results.</p></div>',
                unsafe_allow_html=True,
            )
    with canvas_col:
        render_canvas(preview, actual, people_detected)


page = masthead()
if page == "Image Studio":
    image_studio()
elif page == "Model Results":
    page_heading(
        "02 / EVALUATION", "Model results",
        "Compare the saved CNNs on labelled CIFAR-10 test images.",
    )
    values = accuracies()
    if values:
        with st.container(key="benchmark_scores"):
            columns = st.columns(len(values))
            for column, (name, value) in zip(columns, values.items()):
                column.metric(name, f"{value:.2f}%")
    verified = read_json("verified_evaluation.json")
    checked_rows = []
    current = {}
    for name, details in verified.get("models", {}).items():
        candidate = MODELS / name
        if candidate.is_file() and details.get("checkpoint_sha256") == file_sha256(candidate):
            current[name] = details
            checked_rows.append({"Model file": name, "Test images": details["total"], "Correct": details["correct"], "Accuracy (%)": details["test_accuracy"]})
    comparison, class_results = st.columns([1.55, 1], gap="medium") if values and current else (st.container(), st.container())
    if values:
        with comparison, st.container(border=True):
            st.subheader("Test accuracy")
            chart_data = pd.DataFrame([
                {"Model": name, "Accuracy (%)": value} for name, value in values.items()
            ])
            st.bar_chart(chart_data, x="Model", y="Accuracy (%)", horizontal=True, height=250)
            if "Continued CNN" in values and "Advanced CNN" in values:
                improvement = values["Continued CNN"] - values["Advanced CNN"]
                st.caption(f"Continued CNN gained {improvement:+.2f} percentage points over Advanced CNN.")
    if current:
        with class_results, st.container(border=True):
            selected_check = st.selectbox("Per-class results", list(current), index=len(current) - 1, key="class_results")
            per_class = current[selected_check].get("per_class_accuracy", {})
            st.dataframe(
                pd.DataFrame([{"Class": name, "Accuracy (%)": value} for name, value in per_class.items()]),
                hide_index=True, width="stretch",
                column_config={"Accuracy (%)": st.column_config.ProgressColumn("Accuracy", min_value=0, max_value=100, format="%.2f%%")},
            )
    if checked_rows:
        st.subheader("Verified checkpoints")
        st.dataframe(pd.DataFrame(checked_rows), hide_index=True, width="stretch")
    example_info = read_json("prediction_examples_model.json")
    example_checkpoint = MODELS / example_info.get("checkpoint", "__missing__")
    if example_checkpoint.is_file() and example_info.get("checkpoint_sha256") == file_sha256(example_checkpoint) and (RESULTS / "prediction_examples.png").is_file():
        with st.expander("A fixed random sample of test predictions"):
            st.caption("24 randomly selected test images, including mistakes. Model: " + example_checkpoint.name)
            st.image(str(RESULTS / "prediction_examples.png"), width="stretch")
    for title, filename in [
        ("Advanced CNN experiment", "advanced_evaluation.json"),
        ("Continued CNN experiment", "continued_evaluation.json"),
    ]:
        details = read_json(filename)
        if details:
            with st.expander(title):
                st.json(details)
    original_matrix = RESULTS / "charts" / "confusion_matrix.png"
    if original_matrix.is_file():
        with st.expander("Original NAS confusion matrix"):
            st.caption("This figure describes the original NAS checkpoint.")
            st.image(str(original_matrix), width="stretch")


elif page == "Architecture Lab":
    page_heading(
        "03 / ARCHITECTURE SEARCH", "Architecture lab",
        "Review the candidate CNNs, their validation scores, and the selected configuration.",
    )
    path = RESULTS / "nas_search_results.csv"
    try:
        rankings = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        rankings = pd.DataFrame()
    best = read_json("best_architecture.json")
    candidate_plot, selected_panel = st.columns([1.55, 1], gap="medium") if best else (st.container(), None)
    with candidate_plot, st.container(border=True):
        st.subheader("Candidate validation scores")
        if rankings.empty:
            st.info("No architecture search results are available yet.")
        else:
            st.bar_chart(rankings, x="architecture_id", y="validation_accuracy", x_label="Architecture", y_label="Validation accuracy (%)", height=245)
    if best:
        with selected_panel, st.container(border=True):
            st.subheader("Selected architecture")
            if not rankings.empty:
                st.metric("Architectures evaluated", len(rankings))
            st.json(best.get("best_configuration", best.get("best_config", {})))
    elif not rankings.empty:
        st.metric("Architectures evaluated", len(rankings))
    if not rankings.empty:
        st.subheader("Candidate rankings")
        st.dataframe(rankings, hide_index=True, width="stretch")
    chart_pairs = [
        ("parameter_comparison.png", "training_time_comparison.png"),
        ("accuracy_vs_parameters.png", "validation_accuracy_curves.png"),
    ]
    if any((RESULTS / "charts" / name).is_file() for pair in chart_pairs for name in pair):
        with st.expander("Saved search plots"):
            for pair in chart_pairs:
                columns = st.columns(2)
                for column, name in zip(columns, pair):
                    chart = RESULTS / "charts" / name
                    if chart.is_file():
                        column.image(str(chart), width="stretch")


elif page == "Training History":
    page_heading(
        "04 / TRAINING", "Training history",
        "Compare training and validation accuracy across saved runs.",
    )
    runs = {}
    for title, filename in [
        ("Advanced CNN · initial training", "advanced_history.json"),
        ("Continued CNN · fine-tuning", "continued_history.json"),
    ]:
        history = read_json(filename, [])
        if isinstance(history, list) and history:
            runs[title] = pd.DataFrame(history)
    history_panel, command_panel = st.columns([1.7, 1], gap="medium")
    with history_panel, st.container(border=True):
        if runs:
            run_title = st.selectbox("Training run", list(runs), index=len(runs) - 1)
            frame = runs[run_title]
            if {"epoch", "train_accuracy", "val_accuracy"}.issubset(frame.columns):
                st.line_chart(frame.set_index("epoch")[["train_accuracy", "val_accuracy"]], y_label="Accuracy (%)")
            with st.expander("Epoch details"):
                st.dataframe(frame, hide_index=True, width="stretch")
        else:
            st.info("No saved training runs are available yet.")
    with command_panel, st.container(border=True):
        st.subheader("Continue training")
        st.code(r".\.venv\Scripts\python.exe -m src.continue_training --epochs 10 --batch-size 64", language="powershell")
        st.caption("Checkpoint selection uses the original validation split.")

st.markdown(
    '<footer class="studio-footer"><span>AutoNet Vision Studio</span>'
    '<span>Local inference · CIFAR-10 experiments</span></footer>',
    unsafe_allow_html=True,
)

