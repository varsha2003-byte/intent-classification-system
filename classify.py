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
    """
    Predicts intent using enhanced zero-shot classification with context enrichment.
    
    Args:
        conversation_text: Preprocessed conversation string
        key_signals: Dictionary of detected keyword signals 
    
    Returns:
        tuple: (predicted_intent, detailed_rationale)
    """
    
    # Extract the most recent user messages for focused analysis
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
    
    # Run zero-shot classification with enhanced template
    result = intent_classifier(
        classification_input,
        list(INTENT_DEFINITIONS.keys()),
        hypothesis_template="This customer wants to {}.",
        multi_label=False
    )
    
    # Get predictions
    top_intent = result["labels"][0]
    top_score = result["scores"][0]
    second_intent = result["labels"][1] if len(result["labels"]) > 1 else None
    second_score = result["scores"][1] if len(result["scores"]) > 1 else 0
    
    # Apply rule-based boosting for clear patterns
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
    Generate concise, human-readable rationale in the required format.
    Format: "The user [action] after [context]."
    """
    text_lower = text.lower()
    
    # Intent-specific concise reasoning
    if intent == "Book Appointment":
        if "visit" in text_lower or "viewing" in text_lower:
            return "The user requested a site visit after discussing property requirements."
        elif "appointment" in text_lower or "schedule" in text_lower:
            return "The user requested to schedule an appointment to view the property."
        elif "meet" in text_lower:
            return "The user expressed interest in meeting to discuss the property."
        else:
            return "The user indicated readiness to proceed with a property viewing."
    
    elif intent == "Product Inquiry":
        if "looking for" in text_lower:
            return "The user is exploring available properties and seeking information."
        elif "tell me" in text_lower or "what" in text_lower:
            return "The user asked for detailed information about property features."
        elif "amenities" in text_lower or "features" in text_lower:
            return "The user inquired about property amenities and specifications."
        else:
            return "The user is gathering information about available properties."
    
    elif intent == "Pricing Negotiation":
        if "discount" in text_lower:
            return "The user requested a discount after discussing the property price."
        elif "negotiate" in text_lower:
            return "The user expressed interest in negotiating the property price."
        elif "budget" in text_lower:
            return "The user discussed budget constraints and pricing expectations."
        elif "cheaper" in text_lower or "expensive" in text_lower:
            return "The user indicated the price was high and sought better pricing options."
        else:
            return "The user engaged in price negotiation after reviewing the property."
    
    elif intent == "Support Request":
        if "error" in text_lower or "not working" in text_lower:
            return "The user reported a technical issue and requested assistance."
        elif "problem" in text_lower or "issue" in text_lower:
            return "The user encountered a problem and asked for help resolving it."
        elif "help" in text_lower:
            return "The user requested support to resolve an ongoing issue."
        else:
            return "The user needs assistance with a service-related concern."
    
    elif intent == "Follow-Up":
        if "following up" in text_lower or "follow up" in text_lower:
            return "The user is following up on a previous property inquiry."
        elif "any update" in text_lower or "status" in text_lower:
            return "The user checked the status of their earlier request."
        elif "previous" in text_lower or "last time" in text_lower:
            return "The user referenced a prior conversation and requested an update."
        else:
            return "The user followed up on a previously discussed property or request."
    
    # Fallback
    return f"The user's conversation indicates a {intent.lower()} intent."
