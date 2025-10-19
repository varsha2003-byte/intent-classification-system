from transformers import pipeline
import re

# Intent labels with detailed descriptions for better zero-shot performance
INTENT_DEFINITIONS = {
    "Book Appointment": "scheduling a meeting, requesting a site visit, booking a viewing, arranging an appointment",
    "Product Inquiry": "asking about products, seeking information, exploring options, learning about features",
    "Pricing Negotiation": "discussing price, negotiating cost, requesting discount, talking about budget",
    "Support Request": "reporting a problem, asking for help, fixing an issue, technical support",
    "Follow-Up": "checking previous request status, following up on earlier conversation, asking for updates"
}

# Load model once globally for efficiency
print("Loading intent classification model...")
intent_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1
)
print("Model loaded successfully!")

def predict_intent(conversation_text, key_signals=None):
    
    
    #Extract the most recent user messages for focused analysis
    parts = conversation_text.split("[FINAL USER INTENT]:")
    if len(parts) > 1:
        final_context = parts[-1].strip()
    else:
        final_context = conversation_text
    
    # Use enriched intent labels with descriptions
    enriched_labels = [
        f"{intent}: {desc}" 
        for intent, desc in INTENT_DEFINITIONS.items()
    ]
    
    # Create a focused input for classification
    classification_input = f"Customer's final request: {final_context}"
    
    #Run zero-shot classification with enhanced template
    result = intent_classifier(
        classification_input,
        list(INTENT_DEFINITIONS.keys()),
        hypothesis_template="This customer wants to {}.",
        multi_label=False
    )
    
    #Get predictions
    top_intent = result["labels"][0]
    top_score = result["scores"][0]
    second_intent = result["labels"][1] if len(result["labels"]) > 1 else None
    second_score = result["scores"][1] if len(result["scores"]) > 1 else 0
    
    #Apply rule-based boosting for clear patterns
    adjusted_intent, adjusted_score = apply_pattern_boosting(
        conversation_text, 
        top_intent, 
        top_score,
        result
    )
    
    # Generate intelligent rationale
    rationale = generate_rationale(
        conversation_text,
        adjusted_intent,
        adjusted_score,
        second_intent,
        second_score
    )
    
    return adjusted_intent, rationale

def apply_pattern_boosting(text, predicted_intent, score, full_result):
    
    text_lower = text.lower()
    
    # Define strong signal patterns for each intent
    strong_patterns = {
        "Book Appointment": [
            r'\b(schedule|book|appointment|visit|viewing|meet|tour|site visit|show)\b',
            r'\b(when can|can we|let\'s meet|available for|free on)\b',
            r'\b(tomorrow|today|this week|next week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
        ],
        "Pricing Negotiation": [
            r'\b(price|cost|budget|expensive|cheaper|discount|negotiate|afford)\b',
            r'\b(can you do|offer|reduce|lower|best price|negotiation)\b',
            r'\b(max|maximum|within|under)\b.*\b(aed|usd|dollar|k|thousand)\b'
        ],
        "Support Request": [
            r'\b(problem|issue|error|broken|not working|help|fix|complaint)\b',
            r'\b(refund|cancel|wrong|incorrect|mistake)\b',
            r'\b(payment.*error|gateway.*error|booking.*error)\b'
        ],
        "Product Inquiry": [
            r'\b(what|which|tell me|information|details|available|options)\b',
            r'\b(looking for|interested in|want to know|show me)\b',
            r'\b(features|amenities|specifications|include)\b'
        ],
        "Follow-Up": [
            r'\b(follow.*up|following up|any update|heard back|status)\b',
            r'\b(last time|previous|earlier|last week|last month)\b',
            r'\b(still waiting|any news|any progress)\b'
        ]
    }
    
    # Count pattern matches for each intent
    pattern_scores = {}
    for intent, patterns in strong_patterns.items():
        match_count = sum(1 for pattern in patterns if re.search(pattern, text_lower))
        pattern_scores[intent] = match_count
    
    # Find intent with most pattern matches
    max_pattern_intent = max(pattern_scores, key=pattern_scores.get)
    max_pattern_count = pattern_scores[max_pattern_intent]
    
    # If we have strong pattern evidence (2+ matches), boost that intent
    if max_pattern_count >= 2:
        # Check if pattern-based intent differs from model prediction
        if max_pattern_intent != predicted_intent:
            # If model confidence is low (<0.6), override with pattern-based intent
            if score < 0.6:
                return max_pattern_intent, 0.75  # Boosted confidence
    
    # Check for appointment-specific strong signals (highest priority)
    appointment_signals = re.search(
        r'\b(can we|when can|let\'s|schedule|book|arrange).*\b(visit|viewing|appointment|meet|tour)\b',
        text_lower
    )
    if appointment_signals and predicted_intent != "Book Appointment":
        if score < 0.7:
            return "Book Appointment", 0.80
    
    # Check for negotiation-specific signals
    negotiation_signals = re.search(
        r'\b(can you|offer|give me).*\b(discount|better price|lower|cheaper)\b',
        text_lower
    )
    if negotiation_signals and predicted_intent != "Pricing Negotiation":
        if score < 0.7:
            return "Pricing Negotiation", 0.75
    
    # Return original prediction if no strong override needed
    return predicted_intent, score

def generate_rationale(text, intent, score, second_intent, second_score):
    """
    Generate human-readable rationale based on classification results.
    """
    text_lower = text.lower()
    
    # Confidence level description
    if score > 0.7:
        confidence_text = "confidently classified"
    elif score > 0.55:
        confidence_text = "classified"
    else:
        confidence_text = "tentatively classified"
    
    rationale_parts = []
    
    # Intent-specific reasoning with keyword detection
    if intent == "Book Appointment":
        keywords = find_keywords(text_lower, [
            "visit", "appointment", "schedule", "meet", "viewing", 
            "show", "tour", "site visit", "when can", "tomorrow",
            "this week", "next week", "available"
        ])
        if keywords:
            rationale_parts.append(f"User explicitly requested scheduling/viewing with keywords: {', '.join(keywords[:3])}")
        else:
            rationale_parts.append("Conversation indicates readiness to proceed with in-person meeting")
    
    elif intent == "Product Inquiry":
        keywords = find_keywords(text_lower, [
            "looking for", "tell me", "what", "information", "details", 
            "available", "options", "interested", "features", "amenities"
        ])
        if keywords:
            rationale_parts.append(f"User is seeking information, indicated by: {', '.join(keywords[:3])}")
        else:
            rationale_parts.append("User is exploring options and gathering product information")
    
    elif intent == "Pricing Negotiation":
        keywords = find_keywords(text_lower, [
            "budget", "price", "cost", "expensive", "cheaper", "discount", 
            "negotiate", "afford", "max", "can you do"
        ])
        if keywords:
            rationale_parts.append(f"Discussion centers on pricing/budget: {', '.join(keywords[:3])}")
        else:
            rationale_parts.append("Conversation focused on financial aspects and pricing")
    
    elif intent == "Support Request":
        keywords = find_keywords(text_lower, [
            "problem", "issue", "help", "not working", "error", 
            "complaint", "fix", "broken", "wrong"
        ])
        if keywords:
            rationale_parts.append(f"User needs assistance with: {', '.join(keywords[:3])}")
        else:
            rationale_parts.append("User is seeking help or reporting an issue")
    
    elif intent == "Follow-Up":
        keywords = find_keywords(text_lower, [
            "follow up", "following up", "any update", "status", 
            "heard back", "previous", "earlier", "last time"
        ])
        if keywords:
            rationale_parts.append(f"User is checking status of previous interaction: {', '.join(keywords[:2])}")
        else:
            rationale_parts.append("User is following up on a previous conversation")
    
    # Mention alternative if close
    if second_score > 0.25 and abs(score - second_score) < 0.25:
        rationale_parts.append(f"Note: '{second_intent}' was also considered (score: {second_score:.2f})")
    
    # Combine rationale
    final_rationale = f"The conversation was {confidence_text} as '{intent}' (confidence: {score:.2f}). "
    if rationale_parts:
        final_rationale += " ".join(rationale_parts)
    
    return final_rationale

def find_keywords(text, keywords):
    """Find which keywords from a list appear in the text"""
    found = []
    for kw in keywords:
        if kw in text:
            found.append(kw)
    return found