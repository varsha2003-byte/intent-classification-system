import json
import pandas as pd
from preprocess import preprocess_conversation, extract_key_signals
from classify import predict_intent
import time
from tqdm import tqdm
import sys

def process_conversations(input_file="sample_input.json", output_json="output.json", output_csv="output.csv"):
     
    print(f"Starting Multi-Turn Intent Classification Pipeline")
    print(f"Reading input from: {input_file}")
    
    # Load input data
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            conversations = json.load(f)
        print(f"Loaded {len(conversations)} conversations")
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)
    
    # Process each conversation
    results = []
    print("\nProcessing conversations...")
    
    start_time = time.time()
    
    for conv in tqdm(conversations, desc="Classifying intents"):
        try:
            # Extract conversation ID and messages
            conv_id = conv.get("conversation_id", "unknown")
            messages = conv.get("messages", [])
            
            if not messages:
                print(f"Warning: Empty conversation {conv_id}")
                continue
            
            # Preprocess conversation
            conversation_text = preprocess_conversation(messages)
            
            # Extract key signals for better rationale
            key_signals = extract_key_signals(messages)
            
            # Predict intent
            predicted_intent, rationale = predict_intent(conversation_text, key_signals)
            
            # Store result
            results.append({
                "conversation_id": conv_id,
                "predicted_intent": predicted_intent,
                "rationale": rationale
            })
            
        except Exception as e:
            print(f"Error processing conversation {conv.get('conversation_id', 'unknown')}: {e}")
            continue
    
    elapsed_time = time.time() - start_time
    
    # Save results to JSON
    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nJSON output saved to: {output_json}")
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")
    
    # Save results to CSV
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"CSV output saved to: {output_csv}")
    except Exception as e:
        print(f"Error saving CSV: {e}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("CLASSIFICATION SUMMARY")
    print("="*60)
    print(f"Total conversations processed: {len(results)}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Average time per conversation: {elapsed_time/len(results):.3f} seconds")
    print("\nIntent Distribution:")
    
    df = pd.DataFrame(results)
    intent_counts = df["predicted_intent"].value_counts()
    for intent, count in intent_counts.items():
        percentage = (count / len(results)) * 100
        print(f"  • {intent}: {count} ({percentage:.1f}%)")
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)

def main():
    """Entry point for the application"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Turn Intent Classification for WhatsApp-Style Conversations"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="sample_input.json",
        help="Path to input JSON file (default: sample_input.json)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="output.json",
        help="Path for output JSON file (default: output.json)"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="output.csv",
        help="Path for output CSV file (default: output.csv)"
    )
    
    args = parser.parse_args()
    
    process_conversations(
        input_file=args.input,
        output_json=args.output_json,
        output_csv=args.output_csv
    )

if __name__ == "__main__":
    main()