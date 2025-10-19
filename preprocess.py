import re

def preprocess_conversation(messages):
    """
    Enhanced preprocessing that emphasizes the final user intent.
    The key insight: Final intent is usually in the LAST user message(s).
    """
    if not messages:
        return ""
    
    # Separate user and agent messages
    user_messages = [m["text"] for m in messages if m["sender"] == "user"]
    agent_messages = [m["text"] for m in messages if m["sender"] == "agent"]
    
    if not user_messages:
        return ""
    
    # Clean all messages
    cleaned_user_msgs = [clean_message(msg) for msg in user_messages]
    cleaned_agent_msgs = [clean_message(msg) for msg in agent_messages]
    
    # Strategy 1: Create full conversation context (last 6 messages total)
    all_messages = []
    for msg in messages:
        sender = msg["sender"]
        text = clean_message(msg["text"])
        all_messages.append(f"{sender.capitalize()}: {text}")
    
    # Keep last 6 turns for context
    recent_context = " | ".join(all_messages[-6:])
    
    # Strategy 2: Heavily emphasize the last 2 user messages
    # This is where the final intent usually appears
    last_user_messages = " ".join(cleaned_user_msgs[-2:])
    
    # Combine: full context + emphasized final intent
    final_text = f"{recent_context} [FINAL USER INTENT]: {last_user_messages}"
    
    return final_text

def clean_message(text):
    """
    Clean individual message while preserving important keywords and context.
    """
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove excessive punctuation but keep important ones
    text = re.sub(r'[^\w\s\?\!,.\'-]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip extra spaces
    text = text.strip()
    
    return text

def extract_key_signals(messages):
    """
    Extract key phrases that signal specific intents.
    This provides additional context for classification.
    """
    user_texts = " ".join([m["text"].lower() for m in messages if m["sender"] == "user"])
    
    signals = {
        "appointment": ["visit", "appointment", "schedule", "meet", "viewing", "show", "tour", "when can", "available"],
        "inquiry": ["looking for", "tell me about", "what is", "information", "details", "available", "options"],
        "pricing": ["price", "cost", "budget", "discount", "negotiate", "cheaper", "expensive", "afford"],
        "support": ["problem", "issue", "help", "not working", "error", "complaint", "refund", "fix"],
        "followup": ["follow up", "following up", "any update", "heard back", "status", "previous", "last time"]
    }
    
    detected_signals = {}
    for signal_type, keywords in signals.items():
        detected = [kw for kw in keywords if kw in user_texts]
        if detected:
            detected_signals[signal_type] = detected
    
    return detected_signals