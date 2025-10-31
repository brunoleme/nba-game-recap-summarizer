"""
FIXED GENERATION FUNCTIONS FOR YOUR COLAB NOTEBOOK
Copy and paste these functions to replace the problematic ones in your notebook
"""

def generate_with_model_FIXED(mdl, tok, prmpt, max_new_tokens=256):
    """FIXED generation function with proper parameters and error handling."""
    try:
        # Ensure model is in eval mode
        mdl.eval()
        
        # Tokenize with proper truncation - THIS IS THE KEY FIX
        inputs = tok(
            prmpt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024,  # MUCH SMALLER - this was the main issue
            padding=True
        )
        
        # Move to device
        device = next(mdl.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Check input length
        input_length = inputs['input_ids'].shape[1]
        print(f"🔍 Input length: {input_length} tokens")
        
        # If still too long, truncate more aggressively
        if input_length > 900:
            print(f"⚠️ Input too long ({input_length}), truncating more...")
            inputs = tok(
                prmpt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=800,  # Even more aggressive
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_length = inputs['input_ids'].shape[1]
            print(f"🔍 New input length: {input_length} tokens")
        
        # Generate with BETTER parameters
        with torch.no_grad():
            outputs = mdl.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,  # Added top_k
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
                repetition_penalty=1.1,
                use_cache=True,  # Enable cache
                early_stopping=True,  # Stop at EOS
                num_beams=1  # Use greedy decoding
            )
        
        # Extract ONLY the generated part
        generated_ids = outputs[0][input_length:]
        generated = tok.decode(generated_ids, skip_special_tokens=True)
        
        # Postprocess
        generated = postprocess_text(generated)
        
        print(f"🔍 Generated: {len(generated)} chars, {len(generated.split())} words")
        print(f"🔍 First 100 chars: {generated[:100]}")
        
        return generated.strip()
        
    except Exception as e:
        print(f"❌ Generation error: {e}")
        import traceback
        traceback.print_exc()
        return ""

def build_simple_prompt_FIXED(game_recap):
    """Build a MUCH SHORTER prompt to avoid token overflow."""
    # Truncate recap aggressively
    if len(game_recap) > 600:  # Much shorter
        game_recap = game_recap[:600] + "..."
    
    return f"Summarize this NBA game recap:\n\n{game_recap}\n\nSummary:"

def test_basic_generation():
    """Test basic generation to verify the model works."""
    print("\n" + "="*60)
    print("TESTING BASIC GENERATION")
    print("="*60)
    
    # Very simple test
    test_prompt = "Summarize this game: Lakers beat Warriors 120-115. LeBron scored 30 points.\n\nSummary:"
    
    print(f"Test prompt: {test_prompt}")
    print(f"Prompt tokens: {len(tokenizer.encode(test_prompt))}")
    
    # Test with tuned model
    result = generate_with_model_FIXED(model, tokenizer, test_prompt, max_new_tokens=100)
    
    print(f"Result: '{result}'")
    print(f"Length: {len(result)} chars")
    
    return result

# REPLACE YOUR generate_with_model FUNCTION WITH THIS:
def generate_with_model(mdl, tok, prmpt):
    """REPLACE YOUR ORIGINAL FUNCTION WITH THIS FIXED VERSION"""
    return generate_with_model_FIXED(mdl, tok, prmpt, max_new_tokens=256)

# REPLACE YOUR PROMPT BUILDING WITH THIS:
def build_prompt_for_comparison(game_recap):
    """REPLACE YOUR ORIGINAL PROMPT BUILDING WITH THIS SHORTER VERSION"""
    return build_simple_prompt_FIXED(game_recap)

"""
INSTRUCTIONS FOR YOUR COLAB NOTEBOOK:

1. Replace your generate_with_model function with the FIXED version above
2. Replace your prompt building with the SHORTER version above  
3. Run test_basic_generation() first to verify it works
4. Then run your before/after comparison

The main issues were:
- Input prompts were too long (600-1300+ tokens)
- max_length was too high (2500)
- No proper truncation
- Missing generation parameters

The fixes:
- Much shorter prompts (max 600 chars)
- Lower max_length (1024 tokens)
- Better generation parameters
- Proper error handling
"""

