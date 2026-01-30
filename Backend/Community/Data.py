import os
import json
import pandas as pd
import tarfile
import urllib.request
from collections import Counter
import re

# ---------------------------
# Dataset Paths
# ---------------------------
BASE_DIR = "dataset/raw"
PROCESSED_DIR = "dataset/processed"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# URLs
HATEXPLAIN_URL = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/dataset.json"
SBIC_URL = "https://homes.cs.washington.edu/~msap/social-bias-frames/SBIC.v2.tgz"
ETHOS_DIR = os.path.join(BASE_DIR, "ethos/Ethos-Hate-Speech-Dataset-master")

# ---------------------------
# Content-Based Subcategory Detection
# ---------------------------
RACIAL_ETHNIC_KEYWORDS = [
    # Racial/Ethnic groups
    r'\bblack\b', r'\bwhite\b', r'\basian\b', r'\blatino\b', r'\blatina\b', r'\bhispanic\b',
    r'\bafrican\b', r'\barab\b', r'\bmiddle eastern\b', r'\bindian\b', r'\bchinese\b',
    r'\bjapanese\b', r'\bkorean\b', r'\bnative\b', r'\bindigenous\b', r'\baboriginal\b',
    r'\bpakistani\b', r'\bbangladeshi\b', r'\bvietnamese\b', r'\bthai\b', r'\bfilipino\b',
    r'\bsouth asian\b', r'\beast asian\b', r'\bsoutheast asian\b',
    # Slurs and offensive terms
    r'\bnigger\b', r'\bnigga\b', r'\bchink\b', r'\bspic\b', r'\bwetback\b', r'\bgook\b',
    r'\bbeaner\b', r'\bcoon\b', r'\bpaki\b', r'\brag\s?head\b', r'\bsand\s?nigger\b',
    r'\bkaffir\b', r'\bcoolie\b', r'\bmongo\b',
    # Racial concepts
    r'\brace\b', r'\bracial\b', r'\bethnic\b', r'\bethnicity\b', r'\bminority\b',
    r'\bpeople of color\b', r'\bpoc\b', r'\bcolored\b', r'\bskin color\b',
    # Stereotypes
    r'\bghetto\b', r'\bhood\b', r'\bthug\b', r'\bgangster\b', r'\bmonkey\b', r'\bape\b',
]

RELIGIOUS_KEYWORDS = [
    # Religions
    r'\bmuslim\b', r'\bislam\b', r'\bchristian\b', r'\bjew\b', r'\bjewish\b', r'\bhindu\b',
    r'\bbuddhist\b', r'\bsikh\b', r'\batheist\b', r'\bcatholic\b', r'\bprotestant\b',
    r'\bislamic\b', r'\bjews\b', r'\bmuslims\b', r'\bhindus\b', r'\bchristians\b',
    # Religious terms
    r'\breligion\b', r'\breligious\b', r'\bmosque\b', r'\bchurch\b', r'\bsynagogue\b',
    r'\btemple\b', r'\ballah\b', r'\bgod\b', r'\bjesus\b', r'\bprophet\b', r'\bmohammed\b',
    r'\bquran\b', r'\bbible\b', r'\btorah\b',
    # Slurs
    r'\bkike\b', r'\bheeb\b', r'\braghead\b', r'\bterrorist\b', r'\bisis\b',
    r'\bjihad\b', r'\bsharia\b', r'\binfidel\b', r'\bkafir\b',
]

NATIONALITY_KEYWORDS = [
    # Nationalities
    r'\bmexican\b', r'\bimmigrant\b', r'\brefugee\b', r'\billegal\b', r'\balien\b',
    r'\bforeigner\b', r'\bmigrant\b', r'\bborder\b', r'\bdeportation\b', r'\bdeport\b',
    r'\bamerican\b', r'\bcanadian\b', r'\beuropean\b', r'\bmexicans\b', r'\bimmigrants\b',
    # Immigration terms
    r'\bimmigration\b', r'\billegal alien\b', r'\bundocumented\b', r'\bvisa\b',
    r'\binvader\b', r'\binvasion\b', r'\bgo back\b', r'\bgo home\b', r'\bdeport\b',
    r'\basylee\b', r'\basylum\b',
]

def infer_subcategory_from_text(text):
    """
    Analyze text content to determine if it contains racial/ethnic, religious, or nationality-based hate speech
    Returns: 'racial_ethnic', 'religious', 'nationality', or None
    """
    if pd.isna(text) or not isinstance(text, str):
        return None
    
    text_lower = text.lower()
    
    # Count matches for each category
    racial_matches = sum(1 for pattern in RACIAL_ETHNIC_KEYWORDS if re.search(pattern, text_lower))
    religious_matches = sum(1 for pattern in RELIGIOUS_KEYWORDS if re.search(pattern, text_lower))
    nationality_matches = sum(1 for pattern in NATIONALITY_KEYWORDS if re.search(pattern, text_lower))
    
    # Return the category with most matches (if any)
    max_matches = max(racial_matches, religious_matches, nationality_matches)
    
    if max_matches == 0:
        return None
    
    if racial_matches == max_matches:
        return "racial_ethnic"
    elif religious_matches == max_matches:
        return "religious"
    elif nationality_matches == max_matches:
        return "nationality"
    
    return None

def normalize_subcategory(subcategory_raw, text=""):
    """
    First try to use existing labels, then fall back to text-based inference
    """
    if pd.isna(subcategory_raw):
        subcategory_raw = ""
    
    subcategory = str(subcategory_raw).lower()
    
    # If we have a good existing label, use it
    if subcategory and subcategory not in ["[]", "", "other", "generic", "none", "unknown"]:
        # Check for racial/ethnic
        if any(keyword in subcategory for keyword in ["black", "white", "asian", "latino", "latina", 
                                                       "hispanic", "african", "race", "racial", "ethnic"]):
            return "racial_ethnic"
        
        # Check for religious
        if any(keyword in subcategory for keyword in ["muslim", "islam", "christian", "jewish", "jew",
                                                       "hindu", "buddhist", "religion", "religious"]):
            return "religious"
        
        # Check for nationality
        if any(keyword in subcategory for keyword in ["immigrant", "refugee", "mexican", "nationality",
                                                       "foreigner", "migrant"]):
            return "nationality"
    
    # If no good label, infer from text content
    return infer_subcategory_from_text(text)

# ---------------------------
# Load HateXplain
# ---------------------------
def load_hatexplain_dataset():
    print("🔹 Loading HateXplain dataset...")
    hatexplain_path = os.path.join(BASE_DIR, "hatexplain.json")
    
    if not os.path.exists(hatexplain_path):
        print("   Downloading HateXplain...")
        urllib.request.urlretrieve(HATEXPLAIN_URL, hatexplain_path)

    with open(hatexplain_path, "r") as f:
        data = json.load(f)

    rows = []
    hate_count = 0
    normal_count = 0
    
    for _, post_data in data.items():
        text = " ".join(post_data["post_tokens"])
        annotators = post_data.get("annotators", [])

        labels, targets = [], []
        for ann in annotators:
            if isinstance(ann, dict):
                labels.extend(ann.get("label", []))
                targets.extend(ann.get("target", []))

        if not labels:
            continue

        majority_label = Counter(labels).most_common(1)[0][0]
        majority_target = Counter(targets).most_common(1)[0][0] if targets else "none"

        # Use both existing label and text inference
        subcategory = normalize_subcategory(majority_target, text)
        
        # Keep hate speech samples (with or without subcategory - we'll filter later)
        if "hate" in majority_label:
            hate_count += 1
            rows.append({
                "text": text,
                "label": 1,
                "subcategory": subcategory if subcategory else "unknown",
                "source": "HateXplain"
            })
        elif "normal" in majority_label and normal_count < 10000:  # Limit normal samples
            normal_count += 1
            rows.append({
                "text": text,
                "label": 0,
                "subcategory": "none",
                "source": "HateXplain"
            })

    df = pd.DataFrame(rows)
    print(f"✅ Loaded HateXplain: {len(df)} samples ({hate_count} hate, {normal_count} normal)")
    
    if df.empty:
        return pd.DataFrame(columns=["text", "label", "subcategory", "source"])
    
    return df

# ---------------------------
# Load Social Bias Frames (SBIC)
# ---------------------------
def load_sbic_dataset():
    print("🔹 Loading Social Bias Frames dataset...")
    sbic_tgz = os.path.join(BASE_DIR, "SBIC.v2.tgz")
    sbic_extract_dir = os.path.join(BASE_DIR, "sbic")

    # Check if already extracted
    possible_csv_paths = [
        os.path.join(sbic_extract_dir, "SBIC.v2.agg.trn.csv"),
        os.path.join(sbic_extract_dir, "SBIC.v2", "SBIC.v2.agg.trn.csv"),
        os.path.join(sbic_extract_dir, "SBIC.v2.agg.dev.csv"),
        os.path.join(sbic_extract_dir, "SBIC.v2", "SBIC.v2.agg.dev.csv"),
        os.path.join(sbic_extract_dir, "SBIC.v2.agg.tst.csv"),
        os.path.join(sbic_extract_dir, "SBIC.v2", "SBIC.v2.agg.tst.csv"),
    ]

    csv_path = None
    for path in possible_csv_paths:
        if os.path.exists(path):
            csv_path = path
            break

    if not csv_path:
        print(f"⚠️ SBIC CSV not found. Please ensure dataset is extracted in {sbic_extract_dir}")
        return pd.DataFrame(columns=["text", "label", "subcategory", "source"])

    print(f"   Found CSV at: {csv_path}")
    df = pd.read_csv(csv_path)
    
    rows = []
    for _, row in df.iterrows():
        # Extract text
        text = None
        if "post" in df.columns:
            text = row["post"]
        
        if not text or pd.isna(text):
            continue
        
        # Extract label
        label = 0
        if "offensiveYN" in df.columns:
            label = 1 if str(row["offensiveYN"]).lower() in ["yes", "1.0", "1"] else 0
        elif "offensive" in df.columns:
            label = 1 if row["offensive"] > 0.5 else 0
        
        # Get subcategory from both label and text
        subcategory_raw = ""
        if "targetMinority" in df.columns and not pd.isna(row["targetMinority"]):
            subcategory_raw = str(row["targetMinority"])
        elif "targetCategory" in df.columns and not pd.isna(row["targetCategory"]):
            subcategory_raw = str(row["targetCategory"])
        
        subcategory = normalize_subcategory(subcategory_raw, text)
        
        rows.append({
            "text": text,
            "label": label,
            "subcategory": subcategory if subcategory else "unknown",
            "source": "SBIC"
        })
    
    result_df = pd.DataFrame(rows)
    print(f"✅ Loaded SBIC: {len(result_df)} samples")
    return result_df

# ---------------------------
# Load ETHOS Dataset (Fixed)
# ---------------------------
def load_ethos_dataset():
    print("🔹 Loading ETHOS dataset...")
    possible_paths = [
        os.path.join(ETHOS_DIR, "ethos/ethos_data/Ethos_Dataset_Binary.csv"),
        os.path.join(ETHOS_DIR, "ethos_data/Ethos_Dataset_Binary.csv"),
        os.path.join(BASE_DIR, "ethos/Ethos_Dataset_Binary.csv")
    ]

    ethos_file = next((p for p in possible_paths if os.path.exists(p)), None)
    if not ethos_file:
        print("⚠️ ETHOS dataset not found.")
        return pd.DataFrame(columns=["text", "label", "subcategory", "source"])

    # Try reading with different delimiters and error handling
    try:
        # First, try reading with semicolon delimiter (common in ETHOS)
        df = pd.read_csv(ethos_file, delimiter=';', on_bad_lines='skip', encoding='utf-8')
    except:
        try:
            # Try comma delimiter
            df = pd.read_csv(ethos_file, delimiter=',', on_bad_lines='skip', encoding='utf-8')
        except Exception as e:
            print(f"⚠️ Error reading ETHOS file: {e}")
            return pd.DataFrame(columns=["text", "label", "subcategory", "source"])
    
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    print(f"   ETHOS columns found: {df.columns.tolist()}")

    # Find text and label columns
    text_col = None
    label_col = None
    
    for col in df.columns:
        if "comment" in col or "text" in col:
            text_col = col
        if "hate" in col or "isHate" in col.lower():
            label_col = col

    if not text_col or not label_col:
        print(f"⚠️ ETHOS file missing expected columns → found: {df.columns.tolist()}")
        return pd.DataFrame(columns=["text", "label", "subcategory", "source"])

    rows = []
    skipped = 0
    
    for idx, row in df.iterrows():
        try:
            text = str(row[text_col]).strip()
            
            # Skip if text is empty or too short
            if not text or len(text) < 3 or text.lower() in ['nan', 'none', '']:
                skipped += 1
                continue
            
            # Extract and clean label
            label_val = str(row[label_col]).strip()
            
            # Handle various label formats
            if ';' in label_val:
                # Handle cases where label got merged with text
                parts = label_val.split(';')
                label_val = parts[-1].strip()
            
            # Convert to int
            try:
                label = float(label_val)
                label = int(label)
            except:
                # If can't convert, skip this row
                skipped += 1
                continue
            
            # Ensure label is 0 or 1
            if label not in [0, 1]:
                skipped += 1
                continue
            
            # Infer subcategory from text
            subcategory = infer_subcategory_from_text(text) if label == 1 else "none"
            if not subcategory and label == 1:
                subcategory = "racial_ethnic"  # ETHOS default
            
            rows.append({
                "text": text,
                "label": label,
                "subcategory": subcategory if subcategory else "none",
                "source": "ETHOS"
            })
            
        except Exception as e:
            skipped += 1
            continue
    
    result_df = pd.DataFrame(rows)
    print(f"✅ Loaded ETHOS: {len(result_df)} samples (skipped {skipped} malformed rows)")
    return result_df

# ---------------------------
# Combine All & Balance Dataset
# ---------------------------
if __name__ == "__main__":
    # Load all datasets
    hatexplain_df = load_hatexplain_dataset()
    sbic_df = load_sbic_dataset()
    ethos_df = load_ethos_dataset()

    # Combine dataframes
    dfs_to_combine = []
    for df, name in [(hatexplain_df, "HateXplain"), (sbic_df, "SBIC"), (ethos_df, "ETHOS")]:
        if not df.empty:
            dfs_to_combine.append(df)
        else:
            print(f"⚠️ Skipping {name} - no data loaded")

    if not dfs_to_combine:
        print("❌ No datasets were successfully loaded!")
        exit(1)

    # Combine all datasets
    combined_df = pd.concat(dfs_to_combine, ignore_index=True)
    combined_df.dropna(subset=["text"], inplace=True)
    combined_df = combined_df[combined_df["text"].str.strip() != ""]
    
    # Remove duplicates
    combined_df = combined_df.drop_duplicates(subset=["text"], keep="first")

    print("\n" + "="*60)
    print("📊 DATASET STATISTICS")
    print("="*60)
    print(f"\nTotal samples: {len(combined_df)}")
    
    print("\n📋 By label:")
    label_counts = combined_df["label"].value_counts()
    for label, count in label_counts.items():
        label_name = "Hate Speech" if label == 1 else "Non-Hate"
        print(f"   {label_name}: {count} ({count/len(combined_df)*100:.1f}%)")
    
    print("\n📚 By source:")
    for source, count in combined_df["source"].value_counts().items():
        print(f"   {source}: {count}")
    
    print("\n🏷️ By subcategory (all samples):")
    for subcat, count in combined_df["subcategory"].value_counts().head(10).items():
        print(f"   {subcat}: {count}")
    
    # Separate hate speech and non-hate
    hate_df = combined_df[combined_df["label"] == 1]
    non_hate_df = combined_df[combined_df["label"] == 0]
    
    print("\n" + "="*60)
    print("🎯 HATE SPEECH ANALYSIS")
    print("="*60)
    print(f"\nTotal hate speech samples: {len(hate_df)}")
    print("\nHate speech by subcategory:")
    for subcat, count in hate_df["subcategory"].value_counts().items():
        print(f"   {subcat}: {count} ({count/len(hate_df)*100:.1f}%)")
    
    # Filter hate speech to only keep relevant categories
    hate_categorized = hate_df[hate_df["subcategory"].isin(["racial_ethnic", "religious", "nationality"])]
    hate_unknown = hate_df[hate_df["subcategory"] == "unknown"]
    
    print(f"\n📈 Coverage Analysis:")
    print(f"   Total hate speech: {len(hate_df)}")
    print(f"   With relevant category: {len(hate_categorized)} ({len(hate_categorized)/len(hate_df)*100:.1f}%)")
    print(f"   Unknown category: {len(hate_unknown)} ({len(hate_unknown)/len(hate_df)*100:.1f}%)")
    
    # Balance the dataset
    n_hate = len(hate_categorized)
    
    print("\n" + "="*60)
    print("⚖️ BALANCING DATASET")
    print("="*60)
    
    if len(non_hate_df) > n_hate:
        non_hate_df_balanced = non_hate_df.sample(n=n_hate, random_state=42)
        print(f"Downsampled non-hate from {len(non_hate_df)} to {n_hate}")
    else:
        non_hate_df_balanced = non_hate_df
        print(f"Using all {len(non_hate_df)} non-hate samples")
    
    # Create final balanced dataset
    final_df = pd.concat([hate_categorized, non_hate_df_balanced], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    print(f"\n✅ Final Balanced Dataset:")
    print(f"   Total samples: {len(final_df)}")
    print(f"   Hate speech: {len(final_df[final_df['label'] == 1])} ({len(final_df[final_df['label'] == 1])/len(final_df)*100:.1f}%)")
    print(f"   Non-hate: {len(final_df[final_df['label'] == 0])} ({len(final_df[final_df['label'] == 0])/len(final_df)*100:.1f}%)")
    
    print("\n   Hate speech breakdown by subcategory:")
    hate_breakdown = final_df[final_df['label'] == 1]['subcategory'].value_counts()
    for subcat, count in hate_breakdown.items():
        print(f"   • {subcat}: {count} ({count/len(hate_categorized)*100:.1f}%)")

    # Save datasets
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("\n" + "="*60)
    print("💾 SAVING DATASETS")
    print("="*60)
    
    # Save main balanced dataset
    final_path = os.path.join(PROCESSED_DIR, "community_racial_ethnic_hate_speech.csv")
    final_df.to_csv(final_path, index=False)
    print(f"✅ Saved balanced dataset → {final_path}")
    print(f"   ({len(final_df)} samples)")
    
    # Save hate-only subset
    hate_only_path = os.path.join(PROCESSED_DIR, "hate_speech_only.csv")
    hate_categorized.to_csv(hate_only_path, index=False)
    print(f"✅ Saved hate-only subset → {hate_only_path}")
    print(f"   ({len(hate_categorized)} samples)")
    
    # Save uncategorized for manual review
    if len(hate_unknown) > 0:
        uncategorized_path = os.path.join(PROCESSED_DIR, "uncategorized_hate_speech.csv")
        hate_unknown.to_csv(uncategorized_path, index=False)
        print(f"⚠️ Saved uncategorized samples → {uncategorized_path}")
        print(f"   ({len(hate_unknown)} samples for manual review)")
    
    # Save all combined data (before filtering)
    all_data_path = os.path.join(PROCESSED_DIR, "all_combined_data.csv")
    combined_df.to_csv(all_data_path, index=False)
    print(f"📦 Saved all combined data → {all_data_path}")
    print(f"   ({len(combined_df)} samples)")
    
    print("\n" + "="*60)
    print("✨ PROCESSING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the balanced dataset for fine-tuning")
    print("2. Check uncategorized samples and manually label if needed")
    print("3. Use the hate_speech_only.csv for analysis")
    print("\n🚀 Ready for model fine-tuning!")