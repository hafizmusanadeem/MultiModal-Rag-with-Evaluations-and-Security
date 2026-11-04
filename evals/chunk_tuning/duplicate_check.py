import json
from pathlib import Path
from collections import Counter

def check_duplicate_questions():

    script_dir = Path(__file__).resolve().parent
    json_path = script_dir / "jafza_qa.json"
    
    # Check if the file exists
    if not json_path.exists():
        print(f"❌ Error: Could not find 'jafza_qa.json' in {script_dir}")
        return

    # 2. Load the JSON data
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON file: {e}")
        return
    except Exception as e:
        print(f"❌ An error occurred while reading the file: {e}")
        return

    # 3. Access the 'questions' key
    questions_list = data.get("questions", [])
    if not isinstance(questions_list, list):
        print("❌ Error: The 'questions' key does not contain a list.")
        return

    # NOTE: Change 'question' below if your dictionary key is named differently 
    # (e.g., 'query', 'text', 'prompt', etc.)
    question_key = "question"
    
    seen = set()
    duplicate_details = []
    all_questions = []

    # 4. Scan for duplicates
    for idx, item in enumerate(questions_list):
        if isinstance(item, dict) and question_key in item:
            q_text = item[question_key].strip()
            all_questions.append(q_text)
            if q_text in seen:
                duplicate_details.append((idx, q_text))
            else:
                seen.add(q_text)
        else:
            print(f"⚠️ Warning: Item at index {idx} is either not a dictionary or missing the '{question_key}' key.")

    # 5. Report the results
    print(f"\nTotal questions scanned: {len(all_questions)}")
    
    if duplicate_details:
        counts = Counter([q for _, q in duplicate_details])
        print(f"🔴 Found {len(duplicate_details)} duplicate instance(s):\n")
        
        for q_text, total_count in counts.items():
            # total_count + 1 accounts for the initial appearance plus the duplicates
            print(f"• \"{q_text}\" (Appeared {total_count + 1} times)")
    else:
        print("🟢 Success: No duplicate questions found!")

if __name__ == "__main__":
    check_duplicate_questions()