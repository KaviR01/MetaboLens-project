import os
import re
import json
import sqlite3
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import streamlit as st

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from rapidfuzz import process, fuzz

import base64

import time

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(page_title="Metabolens", layout="centered")


# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

.main-title {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
}

.sub-title {
    color: #666;
    margin-bottom: 1.25rem;
}

.app-card {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 16px;
    background: rgba(255,255,255,0.03);
}

.report-card {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 16px;
    background: rgba(255,255,255,0.03);
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}

.report-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 10px;
}

.report-text {
    line-height: 1.7;
}

.metric-chip {
    display: inline-block;
    padding: 8px 12px;
    border-radius: 999px;
    border: 1px solid rgba(128,128,128,0.3);
    margin: 5px 6px 5px 0;
    font-size: 0.95rem;
}

.rating-green {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(40,167,69,0.18);
    border: 1px solid rgba(40,167,69,0.45);
    font-weight: 700;
}

.rating-amber {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(255,193,7,0.18);
    border: 1px solid rgba(255,193,7,0.45);
    font-weight: 700;
}

.rating-red {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(220,53,69,0.18);
    border: 1px solid rgba(220,53,69,0.45);
    font-weight: 700;
}

.small-muted {
    color: #777;
    font-size: 0.9rem;
}

.auth-wrap {
    max-width: 580px;
    margin: 0 auto;
}

.center-note {
    text-align: center;
    color: #666;
    margin-top: 8px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)


# ============================================
# FIXED CONFIG (NO SIDEBAR)
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PHASE2_CKPT = "best_deeplabv3plus_mnv2_foodseg103.pth"
CLASS_NAMES_TXT = "class_names.txt"
USDA_TRAIN_CSV = "train.csv"

MIN_AREA_FRAC = 0.01
CONF_THRESH = 0.30

DB_PATH = "metabolens.db"


# ============================================
# SESSION STATE
# ============================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "username" not in st.session_state:
    st.session_state.username = None

if "profile_saved_message" not in st.session_state:
    st.session_state.profile_saved_message = False

if "profile_saved_time" not in st.session_state:
    st.session_state.profile_saved_time = 0


# ============================================
# DATABASE FUNCTIONS
# ============================================
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS health_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL UNIQUE,
            full_name TEXT,
            age INTEGER,
            diabetes TEXT,
            cholesterol TEXT,
            goal TEXT,
            dietary_preference TEXT,
            limitations TEXT,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            food_accuracy TEXT,
            portion_accuracy TEXT,
            comments TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            meal_rating TEXT,
            detected_foods_json TEXT,
            totals_json TEXT,
            recommendations_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def register_user(username: str, email: str, password: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO users (username, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            username.strip(),
            email.strip().lower(),
            hash_password(password),
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
        return True, "Registration successful. You can now log in."
    except sqlite3.IntegrityError:
        return False, "Username or email already exists."
    except Exception as e:
        return False, f"Registration failed: {e}"


def login_user(username_or_email: str, password: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM users
        WHERE username = ? OR email = ?
    """, (
        username_or_email.strip(),
        username_or_email.strip().lower()
    ))
    user = cur.fetchone()
    conn.close()

    if user and user["password_hash"] == hash_password(password):
        return True, user
    return False, None


def get_profile(user_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM health_profiles WHERE user_id = ?", (user_id,))
    profile = cur.fetchone()
    conn.close()
    return profile


def save_profile(user_id: int, full_name: str, age, diabetes: str, cholesterol: str,
                 goal: str, dietary_preference: str, limitations: str):
    conn = get_db_connection()
    cur = conn.cursor()

    existing = get_profile(user_id)
    now = datetime.now().isoformat()

    if existing:
        cur.execute("""
            UPDATE health_profiles
            SET full_name = ?, age = ?, diabetes = ?, cholesterol = ?,
                goal = ?, dietary_preference = ?, limitations = ?, updated_at = ?
            WHERE user_id = ?
        """, (
            full_name, age, diabetes, cholesterol,
            goal, dietary_preference, limitations, now, user_id
        ))
    else:
        cur.execute("""
            INSERT INTO health_profiles
            (user_id, full_name, age, diabetes, cholesterol, goal,
             dietary_preference, limitations, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, full_name, age, diabetes, cholesterol,
            goal, dietary_preference, limitations, now
        ))

    conn.commit()
    conn.close()


def save_feedback(user_id: int, food_accuracy: str, portion_accuracy: str, comments: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO feedback (user_id, food_accuracy, portion_accuracy, comments, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        user_id, food_accuracy, portion_accuracy, comments, datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()


def save_analysis_history(user_id: int, meal_rating: str, detected_foods, totals, recommendations):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO analysis_history
        (user_id, meal_rating, detected_foods_json, totals_json, recommendations_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        meal_rating,
        json.dumps(detected_foods),
        json.dumps(totals),
        json.dumps(recommendations),
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()


def get_recent_history(user_id: int, limit: int = 5):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM analysis_history
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (user_id, limit))
    rows = cur.fetchall()
    conn.close()
    return rows


init_db()


# ============================================
# UTILITY FUNCTIONS
# ============================================
def normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = s.replace("_", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def colorize_mask(mask, num_classes, seed=123):
    rng = np.random.default_rng(seed)
    colors = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    return colors[mask]


def overlay_mask(image_rgb, colored_mask, alpha=0.45):
    if colored_mask.shape[:2] != image_rgb.shape[:2]:
        colored_mask = cv2.resize(
            colored_mask,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    out = (
        image_rgb.astype(np.float32) * (1 - alpha)
        + colored_mask.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)
    return out


def get_rating_class(rating_text: str) -> str:
    rt = rating_text.lower()
    if "green" in rt:
        return "rating-green"
    if "amber" in rt:
        return "rating-amber"
    return "rating-red"


# ============================================
# MODEL LOADERS
# ============================================
@st.cache_resource
def load_class_names(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"class_names.txt not found: {path}")

    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.replace(",", " ").split()
            if len(parts) >= 2 and parts[0].isdigit():
                names.append(" ".join(parts[1:]))
            else:
                names.append(s)
    return names


def build_seg_model(num_classes: int):
    return smp.DeepLabV3Plus(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes
    )


@st.cache_resource
def load_seg_model(ckpt_path: str, class_names_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Segmentation checkpoint not found: {ckpt_path}")

    class_names = load_class_names(class_names_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "num_classes" in ckpt:
        num_classes = int(ckpt["num_classes"])
    else:
        num_classes = len(class_names)

    model = build_seg_model(num_classes)

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state" in ckpt:
            state = ckpt["model_state"]
        else:
            state = ckpt
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, class_names, num_classes


@st.cache_resource
def load_usda_df(train_csv_path: str):
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"USDA train.csv not found: {train_csv_path}")

    df = pd.read_csv(train_csv_path)

    needed = [
        "FoodGroup", "Descrip", "Energy_kcal",
        "Protein_g", "Fat_g", "Carb_g", "Sugar_g", "Fiber_g"
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"USDA train.csv missing columns: {missing}")

    df = df.copy()
    df["desc_norm"] = df["Descrip"].astype(str).map(normalize_text)
    return df


# ============================================
# SEGMENTATION
# ============================================
val_tfms = A.Compose([
    A.LongestMaxSize(max_size=512),
    A.PadIfNeeded(
        min_height=512,
        min_width=512,
        border_mode=cv2.BORDER_CONSTANT,
        value=0
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


def predict_probs_and_mask(seg_model, img_rgb):
    out = val_tfms(
        image=img_rgb,
        mask=np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.int64)
    )
    x = out["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = seg_model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred = np.argmax(probs, axis=0).astype(np.int64)

    return pred, probs


def get_detected_foods(pred_mask, probs, class_names, min_area=0.01):
    h, w = pred_mask.shape
    total = h * w
    detected = []
    num_classes = probs.shape[0]

    for cid in range(num_classes):
        label = class_names[cid] if cid < len(class_names) else f"class_{cid}"

        if label.lower() == "background":
            continue

        pix = (pred_mask == cid)
        count = int(pix.sum())
        area_frac = count / total

        if area_frac < min_area:
            continue

        conf = float(probs[cid][pix].mean()) if count else 0.0

        detected.append({
            "class_id": int(cid),
            "label": label,
            "confidence": conf,
            "area_fraction": float(area_frac),
            "pixel_count": count
        })

    detected.sort(key=lambda x: x["area_fraction"], reverse=True)
    return detected


# ============================================
# PORTION ESTIMATION
# ============================================
GRAMS_PER_1PCT = {
    "rice": 12,
    "bean": 8,
    "beans": 8,
    "carrot": 6,
    "shrimp": 18,
    "prawn": 18,
    "default": 10
}


def estimate_grams(label, area_fraction):
    lab = str(label).lower()
    grams_per_pct = GRAMS_PER_1PCT["default"]

    for key, value in GRAMS_PER_1PCT.items():
        if key != "default" and key in lab:
            grams_per_pct = value
            break

    return float(grams_per_pct * (area_fraction * 100.0))


def phase3_portions(detected_foods, conf_thresh=0.30):
    portions = []

    for food in detected_foods:
        if float(food.get("confidence", 0.0)) < conf_thresh:
            continue

        af = float(food["area_fraction"])
        portions.append({
            "label": food["label"],
            "confidence": float(food["confidence"]),
            "area_fraction": af,
            "pixel_count": int(food.get("pixel_count", 0)),
            "portion_score": af,
            "estimated_grams": estimate_grams(food["label"], af)
        })

    portions.sort(key=lambda x: x["portion_score"], reverse=True)
    return portions


# ============================================
# NUTRITION ANALYSIS
# ============================================
def match_usda_grouped(usda_df, label, min_score=60):
    lab = normalize_text(label)

    if "rice" in lab:
        df_sub = usda_df[
            usda_df["FoodGroup"].str.contains("Cereal Grains and Pasta", case=False, na=False)
        ]
        query = "rice cooked"
        df_sub = df_sub[df_sub["desc_norm"].str.contains("rice", na=False)]

    elif "shrimp" in lab or "prawn" in lab:
        df_sub = usda_df[
            usda_df["FoodGroup"].str.contains("Finfish and Shellfish Products", case=False, na=False)
        ]
        query = "shrimp"
        df_sub = df_sub[df_sub["desc_norm"].str.contains("shrimp", na=False)]

    elif "bean" in lab:
        df_sub = usda_df[
            usda_df["FoodGroup"].str.contains("Vegetable", case=False, na=False)
        ]
        query = "beans green"

    elif "carrot" in lab:
        df_sub = usda_df[
            usda_df["FoodGroup"].str.contains("Vegetable", case=False, na=False)
        ]
        query = "carrots raw"

    else:
        df_sub = usda_df
        query = lab

    if len(df_sub) == 0:
        df_sub = usda_df

    choices = df_sub["desc_norm"].tolist()
    match = process.extractOne(query, choices, scorer=fuzz.WRatio)

    if match is None:
        return None, 0.0

    _, score, idx_local = match
    score = float(score)

    if score < min_score:
        return None, score

    row = df_sub.iloc[idx_local]
    per100 = {
        "matched_name": row["Descrip"],
        "food_group": row["FoodGroup"],
        "kcal": safe_float(row["Energy_kcal"]),
        "protein_g": safe_float(row["Protein_g"]),
        "fat_g": safe_float(row["Fat_g"]),
        "carbs_g": safe_float(row["Carb_g"]),
        "sugar_g": safe_float(row["Sugar_g"]),
        "fiber_g": safe_float(row["Fiber_g"]),
    }

    return per100, score


def scale_per100(per100, grams):
    factor = grams / 100.0
    return {
        "kcal": per100["kcal"] * factor,
        "protein_g": per100["protein_g"] * factor,
        "fat_g": per100["fat_g"] * factor,
        "carbs_g": per100["carbs_g"] * factor,
        "sugar_g": per100["sugar_g"] * factor,
        "fiber_g": per100["fiber_g"] * factor,
    }


def phase4_nutrition(usda_df, portions):
    items = []
    totals = {
        "kcal": 0,
        "protein_g": 0,
        "fat_g": 0,
        "carbs_g": 0,
        "sugar_g": 0,
        "fiber_g": 0
    }

    for portion in portions:
        grams = float(portion.get("estimated_grams", 0.0))
        per100, score = match_usda_grouped(usda_df, portion["label"], min_score=60)

        if per100 is None:
            items.append({
                "label": portion["label"],
                "confidence": float(portion.get("confidence", 1.0)),
                "grams": grams,
                "match_status": "no_match",
                "match_score": score
            })
            continue

        scaled = scale_per100(per100, grams)

        items.append({
            "label": portion["label"],
            "confidence": float(portion.get("confidence", 1.0)),
            "grams": grams,
            "usda_match": per100["matched_name"],
            "food_group": per100["food_group"],
            "match_score": score,
            "nutrition": scaled
        })

        for key in totals:
            totals[key] += float(scaled[key])

    return {"items": items, "totals": totals}


# ============================================
# PERSONALIZED HEALTH INSIGHTS
# ============================================
RULES = {
    "net_carbs_high": 60,
    "net_carbs_very_high": 80,
    "sugar_high": 25,
    "fiber_good": 8,
    "fiber_chol_good": 10,
    "fat_high": 25,
    "protein_low": 20
}


def profile_to_dict(profile_row):
    if not profile_row:
        return {
            "full_name": "",
            "age": None,
            "diabetes": "No",
            "cholesterol": "No",
            "goal": "Maintain health",
            "dietary_preference": "No specific preference",
            "limitations": ""
        }

    return {
        "full_name": profile_row["full_name"] or "",
        "age": profile_row["age"],
        "diabetes": profile_row["diabetes"] or "No",
        "cholesterol": profile_row["cholesterol"] or "No",
        "goal": profile_row["goal"] or "Maintain health",
        "dietary_preference": profile_row["dietary_preference"] or "No specific preference",
        "limitations": profile_row["limitations"] or ""
    }


def phase5_rules(phase4, profile):
    totals = phase4["totals"]
    items = phase4["items"]

    kcal = safe_float(totals.get("kcal", 0))
    carbs = safe_float(totals.get("carbs_g", 0))
    fiber = safe_float(totals.get("fiber_g", 0))
    sugar = safe_float(totals.get("sugar_g", 0))
    protein = safe_float(totals.get("protein_g", 0))
    fat = safe_float(totals.get("fat_g", 0))
    net_carbs = max(carbs - fiber, 0.0)

    diabetes = str(profile.get("diabetes", "No")).lower() == "yes"
    cholesterol = str(profile.get("cholesterol", "No")).lower() == "yes"
    goal = profile.get("goal", "Maintain health")
    dietary_preference = profile.get("dietary_preference", "No specific preference")
    limitations = profile.get("limitations", "")

    recs = []
    scores = {"blood_sugar": 0, "cholesterol": 0, "goal_alignment": 0}

    recs.append({
        "type": "info",
        "message": (
            f"Meal summary: {kcal:.0f} kcal | Net carbs: {net_carbs:.1f} g | "
            f"Protein: {protein:.1f} g | Fiber: {fiber:.1f} g | Fat: {fat:.1f} g | Sugar: {sugar:.1f} g"
        )
    })

    # --------------------------------
    # Diabetes-specific recommendations
    # --------------------------------
    if diabetes:
        if net_carbs >= RULES["net_carbs_very_high"]:
            recs.append({
                "type": "warning",
                "message": "For your diabetes profile, this meal has very high net carbs and may cause a strong blood sugar rise."
            })
            recs.append({
                "type": "recommendation",
                "message": "Reduce rice or other high-carb foods and add more vegetables and lean protein."
            })
            scores["blood_sugar"] += 2

        elif net_carbs >= RULES["net_carbs_high"]:
            recs.append({
                "type": "warning",
                "message": "For your diabetes profile, this meal has high net carbs."
            })
            recs.append({
                "type": "recommendation",
                "message": "Balance this meal with more fiber and protein to improve blood sugar control."
            })
            scores["blood_sugar"] += 1

        else:
            recs.append({
                "type": "positive",
                "message": "For your diabetes profile, the net carbs are at a more manageable level."
            })
            scores["blood_sugar"] -= 1

        if sugar >= RULES["sugar_high"]:
            recs.append({
                "type": "warning",
                "message": "For your diabetes profile, the sugar content is also high."
            })
            scores["blood_sugar"] += 1

    else:
        if net_carbs >= RULES["net_carbs_high"]:
            recs.append({
                "type": "warning",
                "message": "This meal is high in net carbs."
            })
        else:
            recs.append({
                "type": "positive",
                "message": "Net carbs are moderate."
            })

    # --------------------------------
    # Fiber advice
    # --------------------------------
    if fiber >= RULES["fiber_good"]:
        recs.append({
            "type": "positive",
            "message": f"Fiber is good ({fiber:.1f} g), which helps improve fullness and support better glucose control."
        })
        if diabetes:
            scores["blood_sugar"] -= 1
    else:
        recs.append({
            "type": "recommendation",
            "message": "Increase fiber by adding vegetables, legumes, or whole grains."
        })

    # --------------------------------
    # Cholesterol-specific recommendations
    # --------------------------------
    if cholesterol:
        if fat >= RULES["fat_high"]:
            recs.append({
                "type": "warning",
                "message": "For your cholesterol profile, total fat is high. Choose leaner and less oily foods."
            })
            recs.append({
                "type": "recommendation",
                "message": "Reduce fried foods and prefer vegetables, beans, and lean protein."
            })
            scores["cholesterol"] += 1
        else:
            recs.append({
                "type": "positive",
                "message": "For your cholesterol profile, total fat is not high."
            })

        if fiber >= RULES["fiber_chol_good"]:
            recs.append({
                "type": "positive",
                "message": "Higher fiber intake supports better cholesterol control."
            })
            scores["cholesterol"] -= 1
        else:
            recs.append({
                "type": "recommendation",
                "message": "Add more soluble-fiber foods such as oats, vegetables, and legumes."
            })
            scores["cholesterol"] += 0.5

    # --------------------------------
    # Goal-based recommendations
    # --------------------------------
    if goal == "Weight loss":
        if kcal > 600:
            recs.append({
                "type": "recommendation",
                "message": "This meal may be too calorie-dense for a weight loss goal. Consider a smaller portion."
            })
            scores["goal_alignment"] += 1
        else:
            recs.append({
                "type": "positive",
                "message": "This meal is more aligned with a weight loss goal."
            })
            scores["goal_alignment"] -= 1

    elif goal == "Weight gain":
        if protein < RULES["protein_low"]:
            recs.append({
                "type": "recommendation",
                "message": "For weight gain, consider increasing protein and overall meal size."
            })
            scores["goal_alignment"] += 1
        else:
            recs.append({
                "type": "positive",
                "message": "Protein content supports your weight gain goal."
            })
            scores["goal_alignment"] -= 1

    else:
        recs.append({
            "type": "info",
            "message": "This recommendation is balanced for general health maintenance."
        })

    # --------------------------------
    # Preference-based recommendations
    # --------------------------------
    if dietary_preference == "Low sugar" and sugar > RULES["sugar_high"]:
        recs.append({
            "type": "warning",
            "message": "This meal is not well aligned with your low-sugar preference."
        })

    if dietary_preference == "Low fat" and fat > RULES["fat_high"]:
        recs.append({
            "type": "warning",
            "message": "This meal is not well aligned with your low-fat preference."
        })

    if dietary_preference == "High protein" and protein < RULES["protein_low"]:
        recs.append({
            "type": "recommendation",
            "message": "This meal is relatively low in protein for your high-protein preference."
        })

    if limitations.strip():
        recs.append({
            "type": "info",
            "message": f"User limitation noted: {limitations}"
        })

    # --------------------------------
    # Food-specific recommendations
    # --------------------------------
    food_recs = []
    for item in items:
        label = str(item.get("label", ""))
        lab = label.lower()
        grams = safe_float(item.get("grams", 0.0))

        if "rice" in lab:
            if diabetes:
                food_recs.append({
                    "food": label,
                    "type": "warning",
                    "message": "Reduce the rice portion or replace part of it with vegetables for better blood sugar control."
                })
            else:
                food_recs.append({
                    "food": label,
                    "type": "recommendation",
                    "message": "Rice portion may be reduced if you want a lighter meal."
                })

        if any(v in lab for v in ["bean", "beans", "carrot", "vegetable"]):
            food_recs.append({
                "food": label,
                "type": "positive",
                "message": "This food supports better fiber intake."
            })

        if any(p in lab for p in ["shrimp", "fish", "chicken", "egg"]):
            if cholesterol:
                food_recs.append({
                    "food": label,
                    "type": "positive",
                    "message": "This food can be a better protein choice if prepared with less oil."
                })
            else:
                food_recs.append({
                    "food": label,
                    "type": "positive",
                    "message": "This food contributes useful protein."
                })

        if grams > 300:
            food_recs.append({
                "food": label,
                "type": "recommendation",
                "message": "A large portion was detected. Consider reducing serving size."
            })

    total_risk = float(scores["blood_sugar"]) + float(scores["cholesterol"]) + float(scores["goal_alignment"])

    if total_risk >= 2:
        rating = "RED (High concern)"
    elif total_risk >= 1:
        rating = "AMBER (Moderate concern)"
    else:
        rating = "GREEN (Better fit)"

    return {
        "meal_rating": rating,
        "scores": scores,
        "totals_used": {
            "kcal": kcal,
            "carbs_g": carbs,
            "fiber_g": fiber,
            "net_carbs_g": net_carbs,
            "sugar_g": sugar,
            "protein_g": protein,
            "fat_g": fat
        },
        "profile_used": profile,
        "meal_level_recommendations": recs,
        "food_level_recommendations": food_recs
    }


# ============================================
# AUTH UI
# ============================================
def show_auth_page():
    st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size: 28px; font-weight: 800; display:flex; align-items:center; gap:10px;">
        <img src="data:image/png;base64,{get_base64_image('logo.png')}" 
            style="width:32px; height:32px; border-radius:6px;">
        Metabolens
    </div>
    """, unsafe_allow_html=True)
    

    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.subheader("Login")
        login_user_input = st.text_input("Username or Email", key="login_user")
        login_password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login", use_container_width=True):
            ok, user = login_user(login_user_input, login_password)
            if ok:
                st.session_state.logged_in = True
                st.session_state.user_id = user["id"]
                st.session_state.username = user["username"]
                st.success("Login successful.")
                st.rerun()
            else:
                st.error("Invalid username/email or password.")
        st.markdown('</div>', unsafe_allow_html=True)

    with register_tab:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.subheader("Create Account")
        reg_username = st.text_input("Username", key="reg_username")
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_pass")
        reg_password2 = st.text_input("Confirm Password", type="password", key="reg_pass2")

        if st.button("Register", use_container_width=True):
            if not reg_username.strip() or not reg_email.strip() or not reg_password.strip():
                st.error("Please fill in all fields.")
            elif reg_password != reg_password2:
                st.error("Passwords do not match.")
            elif len(reg_password) < 6:
                st.error("Password should be at least 6 characters.")
            else:
                ok, msg = register_user(reg_username, reg_email, reg_password)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="center-note">Create an account to save your profile, feedback, and meal analysis history.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================
# PROFILE UI
# ============================================
def show_profile_tab(user_id: int):
    profile_row = get_profile(user_id)
    profile = profile_to_dict(profile_row)

    st.subheader("Personal Health Profile")
    st.caption("Create and maintain your health profile so the final recommendation becomes personalized.")

    with st.form("profile_form"):
        full_name = st.text_input("Full Name", value=profile["full_name"])
        age = st.number_input("Age", min_value=1, max_value=120, value=int(profile["age"]) if profile["age"] else 21)
        diabetes = st.selectbox("Do you have diabetes?", ["No", "Yes"], index=1 if profile["diabetes"] == "Yes" else 0)
        cholesterol = st.selectbox("Do you have cholesterol concerns?", ["No", "Yes"], index=1 if profile["cholesterol"] == "Yes" else 0)

        goal_options = ["Maintain health", "Weight loss", "Weight gain"]
        goal_index = goal_options.index(profile["goal"]) if profile["goal"] in goal_options else 0
        goal = st.selectbox("Dietary Goal", goal_options, index=goal_index)

        pref_options = ["No specific preference", "Low sugar", "Low fat", "High protein"]
        pref_index = pref_options.index(profile["dietary_preference"]) if profile["dietary_preference"] in pref_options else 0
        dietary_preference = st.selectbox("Dietary Preference", pref_options, index=pref_index)

        limitations = st.text_area("Dietary Limitations / Notes", value=profile["limitations"])

        submitted = st.form_submit_button("Save Health Profile", use_container_width=True)
        if submitted:
            is_new_profile = profile_row is None

            save_profile(
                user_id=user_id,
                full_name=full_name,
                age=int(age),
                diabetes=diabetes,
                cholesterol=cholesterol,
                goal=goal,
                dietary_preference=dietary_preference,
                limitations=limitations
            )

            success_placeholder = st.empty()

            if is_new_profile:
                success_placeholder.success("Profile saved successfully.")
            else:
                success_placeholder.success("Profile updated successfully.")

            time.sleep(3)
            success_placeholder.empty()


# ============================================
# MAIN APP
# ============================================
def show_main_app():
    st.markdown(f"""
    <div style="
    padding: 40px 20px;
    border-radius: 16px;
    background: linear-gradient(135deg, #1e1e2f, #2a2a40);
    margin-bottom: 10px;
    ">

    <div style="font-size: 26px; font-weight: 800; display:flex; align-items:center; gap:8px;">
    <img src="data:image/png;base64,{get_base64_image('logo.png')}" 
         style="width:30px; height:30px; border-radius:6px;">
    Metabolens
    </div>

    <div style="margin-top: 6px; font-size: 15px; color: #bbb;">
    Welcome, <b>{st.session_state.username if st.session_state.username else "User"}</b>
    </div>

    <div style="margin-top: 8px; font-size: 14px; color: #aaa;">
    Analyze your meal and get personalized health insights.
    </div>

    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([5, 1])
    with col_b:
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()

    user_profile_row = get_profile(st.session_state.user_id)
    user_profile = profile_to_dict(user_profile_row)

    main_tabs = st.tabs([
        "Profile",
        "Analyze Meal",
        "History",
        "Feedback"
    ])

    # ----------------------------------------
    # PROFILE TAB
    # ----------------------------------------
    with main_tabs[0]:
        show_profile_tab(st.session_state.user_id)

    # ----------------------------------------
    # ANALYZE MEAL TAB
    # ----------------------------------------
    with main_tabs[1]:
        st.subheader("Meal Analysis")
        st.caption("The system detects food regions, estimates portions, calculates nutrition, and displays a personalized recommendation.")

        uploaded = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

        if uploaded is None:
            st.info("Upload an image to start.")
        else:
            pil_img = Image.open(uploaded).convert("RGB")
            img_rgb = np.array(pil_img)

            st.image(pil_img, caption="Uploaded Image", width=350)

            if st.button("Analyze Meal", type="primary", use_container_width=True):
                errors = []

                if not os.path.exists(PHASE2_CKPT):
                    errors.append(f"Segmentation checkpoint not found: {PHASE2_CKPT}")

                if not os.path.exists(CLASS_NAMES_TXT):
                    errors.append(f"class_names.txt not found: {CLASS_NAMES_TXT}")

                if not os.path.exists(USDA_TRAIN_CSV):
                    errors.append(f"USDA train.csv not found: {USDA_TRAIN_CSV}")

                if errors:
                    st.error("Fix these file issues first:\n\n- " + "\n- ".join(errors))
                    return

                with st.spinner("Loading models and data..."):
                    seg_model, class_names, num_classes = load_seg_model(PHASE2_CKPT, CLASS_NAMES_TXT)
                    usda_df = load_usda_df(USDA_TRAIN_CSV)

                with st.spinner("Analyzing your meal..."):
                    pred_mask, probs = predict_probs_and_mask(seg_model, img_rgb)
                    detected_foods = get_detected_foods(
                        pred_mask, probs, class_names, min_area=MIN_AREA_FRAC
                    )

                    colored = colorize_mask(pred_mask, num_classes=num_classes, seed=123)
                    overlay = overlay_mask(img_rgb, colored, alpha=0.45)

                    portions = phase3_portions(detected_foods, conf_thresh=CONF_THRESH)
                    phase4 = phase4_nutrition(usda_df, portions)
                    phase5 = phase5_rules(phase4, user_profile)

                save_analysis_history(
                    user_id=st.session_state.user_id,
                    meal_rating=phase5["meal_rating"],
                    detected_foods=portions,
                    totals=phase5["totals_used"],
                    recommendations={
                        "meal_level_recommendations": phase5["meal_level_recommendations"],
                        "food_level_recommendations": phase5["food_level_recommendations"]
                    }
                )

                st.success("Analysis completed successfully.")

                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Segmentation",
                    "Portion Estimation",
                    "Nutrition Summary",
                    "Health Insights",
                    "Final Report"
                ])

                # ----------------------------
                # SEGMENTATION
                # ----------------------------
                with tab1:
                    st.subheader("Segmentation Results")
                    st.image(overlay, caption="Segmentation Overlay", width=350)

                    if detected_foods:
                        df_seg = pd.DataFrame(detected_foods)[
                            ["label", "confidence", "area_fraction", "pixel_count"]
                        ].copy()
                        df_seg["confidence"] = df_seg["confidence"].map(lambda x: round(x * 100, 2))
                        df_seg["area_fraction"] = df_seg["area_fraction"].map(lambda x: round(x, 4))
                        df_seg.rename(columns={
                            "label": "Food",
                            "confidence": "Confidence (%)",
                            "area_fraction": "Area Fraction",
                            "pixel_count": "Pixel Count"
                        }, inplace=True)
                        st.dataframe(df_seg, use_container_width=True)
                    else:
                        st.warning("No foods detected.")

                # ----------------------------
                # PORTION ESTIMATION
                # ----------------------------
                with tab2:
                    st.subheader("Estimated Portions")

                    if portions:
                        df_portions = pd.DataFrame(portions)[
                            ["label", "confidence", "area_fraction", "estimated_grams"]
                        ].copy()
                        df_portions["confidence"] = df_portions["confidence"].map(lambda x: round(x * 100, 2))
                        df_portions["area_fraction"] = df_portions["area_fraction"].map(lambda x: round(x, 4))
                        df_portions["estimated_grams"] = df_portions["estimated_grams"].map(lambda x: round(x, 2))
                        df_portions.rename(columns={
                            "label": "Food",
                            "confidence": "Confidence (%)",
                            "area_fraction": "Area Fraction",
                            "estimated_grams": "Estimated Grams"
                        }, inplace=True)
                        st.dataframe(df_portions, use_container_width=True)
                    else:
                        st.warning("No portions produced.")

                # ----------------------------
                # NUTRITION SUMMARY
                # ----------------------------
                with tab3:
                    st.subheader("Nutrition Summary")
                    totals = phase4.get("totals", {})

                    if totals:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Calories", f"{totals.get('kcal', 0):.2f} kcal")
                        c2.metric("Protein", f"{totals.get('protein_g', 0):.2f} g")
                        c3.metric("Fat", f"{totals.get('fat_g', 0):.2f} g")

                        c4, c5, c6 = st.columns(3)
                        c4.metric("Carbohydrates", f"{totals.get('carbs_g', 0):.2f} g")
                        c5.metric("Sugar", f"{totals.get('sugar_g', 0):.2f} g")
                        c6.metric("Fiber", f"{totals.get('fiber_g', 0):.2f} g")

                    st.markdown("#### Food Items")
                    if phase4.get("items"):
                        readable_items = []

                        for item in phase4["items"]:
                            nutrition = item.get("nutrition", {})
                            readable_items.append({
                                "Food": item.get("label", "-"),
                                "Estimated grams": round(item.get("grams", 0), 2),
                                "Matched USDA food": item.get("usda_match", "No match"),
                                "Calories (kcal)": round(nutrition.get("kcal", 0), 2),
                                "Protein (g)": round(nutrition.get("protein_g", 0), 2),
                                "Fat (g)": round(nutrition.get("fat_g", 0), 2),
                                "Carbs (g)": round(nutrition.get("carbs_g", 0), 2),
                                "Sugar (g)": round(nutrition.get("sugar_g", 0), 2),
                                "Fiber (g)": round(nutrition.get("fiber_g", 0), 2),
                            })

                        st.dataframe(pd.DataFrame(readable_items), use_container_width=True)
                    else:
                        st.info("No nutrition results available.")

                # ----------------------------
                # HEALTH INSIGHTS
                # ----------------------------
                with tab4:
                    st.subheader("Health Insights & Recommendations")
                    st.markdown("#### Profile used for personalization")
                    st.write(f"**Diabetes:** {user_profile['diabetes']}")
                    st.write(f"**Cholesterol concern:** {user_profile['cholesterol']}")
                    st.write(f"**Goal:** {user_profile['goal']}")
                    st.write(f"**Dietary preference:** {user_profile['dietary_preference']}")
                    st.write(f"**Limitations:** {user_profile['limitations'] if user_profile['limitations'] else 'None'}")

                    st.markdown(f"### Meal Rating: **{phase5['meal_rating']}**")

                    st.markdown("#### Meal-level recommendations")
                    for rec in phase5["meal_level_recommendations"]:
                        rec_type = rec["type"].lower()
                        msg = rec["message"]

                        if rec_type == "warning":
                            st.warning(msg)
                        elif rec_type == "positive":
                            st.success(msg)
                        elif rec_type == "recommendation":
                            st.info("✅ " + msg)
                        else:
                            st.info(msg)

                    st.markdown("#### Food-specific advice")
                    if phase5["food_level_recommendations"]:
                        for food_rec in phase5["food_level_recommendations"]:
                            rec_type = food_rec["type"].lower()
                            msg = f"{food_rec['food']}: {food_rec['message']}"

                            if rec_type == "warning":
                                st.warning(msg)
                            elif rec_type == "positive":
                                st.success(msg)
                            else:
                                st.info(msg)
                    else:
                        st.info("No food-specific recommendations generated.")

                # ----------------------------
                # FINAL REPORT - CARDS
                # ----------------------------
                with tab5:
                    st.subheader("Final Meal Report")

                    totals_used = phase5["totals_used"]
                    rating_class = get_rating_class(phase5["meal_rating"])

                    st.markdown(f"""
                    <div class="report-card">
                        <div class="report-title">Meal Rating</div>
                        <div class="{rating_class}">{phase5["meal_rating"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="report-card">
                        <div class="report-title">Personal Health Profile Used</div>
                        <div class="report-text">
                            <span class="metric-chip"><b>Diabetes:</b> {user_profile['diabetes']}</span>
                            <span class="metric-chip"><b>Cholesterol:</b> {user_profile['cholesterol']}</span>
                            <span class="metric-chip"><b>Goal:</b> {user_profile['goal']}</span>
                            <span class="metric-chip"><b>Preference:</b> {user_profile['dietary_preference']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="report-card">
                        <div class="report-title">Meal Summary</div>
                        <div class="report-text">
                            <span class="metric-chip"><b>Calories:</b> {totals_used.get('kcal', 0):.2f} kcal</span>
                            <span class="metric-chip"><b>Carbohydrates:</b> {totals_used.get('carbs_g', 0):.2f} g</span>
                            <span class="metric-chip"><b>Fiber:</b> {totals_used.get('fiber_g', 0):.2f} g</span>
                            <span class="metric-chip"><b>Net Carbs:</b> {totals_used.get('net_carbs_g', 0):.2f} g</span>
                            <span class="metric-chip"><b>Sugar:</b> {totals_used.get('sugar_g', 0):.2f} g</span>
                            <span class="metric-chip"><b>Protein:</b> {totals_used.get('protein_g', 0):.2f} g</span>
                            <span class="metric-chip"><b>Fat:</b> {totals_used.get('fat_g', 0):.2f} g</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    detected_foods_html = ""
                    if portions:
                        for portion in portions:
                            detected_foods_html += (
                                f"• <b>{portion['label']}</b> — "
                                f"{portion['estimated_grams']:.2f} g "
                                f"(confidence: {portion['confidence'] * 100:.1f}%)<br>"
                            )
                    else:
                        detected_foods_html = "No foods were confidently detected."

                    st.markdown(f"""
                    <div class="report-card">
                        <div class="report-title">Detected Foods</div>
                        <div class="report-text">{detected_foods_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    meal_recs_html = ""
                    for rec in phase5["meal_level_recommendations"]:
                        meal_recs_html += f"• {rec['message']}<br>"

                    if not meal_recs_html:
                        meal_recs_html = "No recommendations available."

                    st.markdown(f"""
                    <div class="report-card">
                        <div class="report-title">Personalized Recommendations</div>
                        <div class="report-text">{meal_recs_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    food_advice_html = ""
                    if phase5["food_level_recommendations"]:
                        for food_rec in phase5["food_level_recommendations"]:
                            food_advice_html += (
                                f"• <b>{food_rec['food']}</b>: {food_rec['message']}<br>"
                            )
                    else:
                        food_advice_html = "No food-specific advice available."

                    st.markdown(f"""
                    <div class="report-card">
                        <div class="report-title">Food-specific Advice</div>
                        <div class="report-text">{food_advice_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ----------------------------------------
    # HISTORY TAB
    # ----------------------------------------
    with main_tabs[2]:
        st.subheader("Recent Analysis History")
        st.caption("Previously saved meal analyses for the logged-in user.")

        history_rows = get_recent_history(st.session_state.user_id, limit=5)
        if not history_rows:
            st.info("No analysis history found yet.")
        else:
            for row in history_rows:
                totals = json.loads(row["totals_json"])
                foods = json.loads(row["detected_foods_json"])

                foods_text = ", ".join([f.get("label", "-") for f in foods]) if foods else "No foods"
                st.markdown(f"""
                <div class="report-card">
                    <div class="report-title">{row["meal_rating"]}</div>
                    <div class="report-text">
                        <b>Date:</b> {row["created_at"]}<br>
                        <b>Foods:</b> {foods_text}<br>
                        <b>Calories:</b> {safe_float(totals.get("kcal", 0)):.2f} kcal<br>
                        <b>Net Carbs:</b> {safe_float(totals.get("net_carbs_g", 0)):.2f} g
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ----------------------------------------
    # FEEDBACK TAB
    # ----------------------------------------
    with main_tabs[3]:
        st.subheader("User Feedback")
        st.caption("The system should support users in giving feedback about whether the food and portion size were accurately recognized.")

        with st.form("feedback_form"):
            food_accuracy = st.radio(
                "Was the detected food accurate?",
                ["Accurate", "Partly Accurate", "Not Accurate"]
            )

            portion_accuracy = st.radio(
                "Was the portion size accurate?",
                ["Accurate", "Too High", "Too Low"]
            )

            comments = st.text_area("Additional Comments")

            submitted_feedback = st.form_submit_button("Submit Feedback", use_container_width=True)
            if submitted_feedback:
                save_feedback(
                    user_id=st.session_state.user_id,
                    food_accuracy=food_accuracy,
                    portion_accuracy=portion_accuracy,
                    comments=comments
                )
                st.success("Feedback submitted successfully.")


# ============================================
# APP ENTRY
# ============================================
if not st.session_state.logged_in:
    show_auth_page()
else:
    show_main_app()