# Judge Scoring Fix - December 30, 2025

## Problem
The LLM judge was giving **incorrect accuracy scores** for "this or that" questions where the human and AI chose different options (or sometimes even when they chose the same option).

### Specific Issues:
1. **Different choices getting high scores**: When AI chose "Work" and Human chose "Mummy", the accuracy was 83% instead of ~0-23%
2. **Same choices getting low scores**: When both AI and Human chose "Mummy", the score was 23% instead of ~100%

## Root Causes

### Cause 1: Logic Override Bug
**File**: `engine/judge_handler.py` (lines 224-228)

The code was **forcing** preference_alignment to 100 whenever intent_match was YES:

```python
# WRONG CODE (removed):
if result["intent_match"]:
    if result["breakdown"]["preference_alignment"] < 100:
        result["breakdown"]["preference_alignment"] = 100
```

**The Problem**: 
- LLM would correctly set `preference_alignment: 0` (different choices)
- But if it mistakenly set `intent_match: YES`, the code would **override** the preference to 100
- This resulted in high scores for mismatched responses

**The Fix**:
```python
# CORRECT CODE:
# Derive intent_match FROM preference score (not vice versa)
p = result["breakdown"]["preference_alignment"]
if p >= 70:
    result["intent_match"] = True
else:
    result["intent_match"] = False
```

Now `intent_match` is **derived from** the preference score, not the other way around.

---

### Cause 2: Unclear LLM Prompt
**File**: `engine/judge_handler.py` (_create_judge_prompt method)

The original prompt was confusing the LLM. It would sometimes:
- Say "they chose the same option" but give PREFERENCE: 0
- Give inconsistent INTENT_MATCH values

**The Fix**:
Created a **much clearer prompt** with:
1. ✅ **Comparison table** showing explicit examples
2. ✅ **Clear binary logic**: Same choice = 100, Different choice = 0
3. ✅ **No ambiguity**: Direct instructions with examples

```
=== COMPARISON TABLE ===
| Example          | AI Says | Human Says | PREFERENCE | INTENT_MATCH |
|------------------|---------|------------|------------|--------------|
| Same choice      | "Work"  | "Work"     | 100        | YES          |
| Different choice | "Work"  | "Mummy"    | 0          | NO           |
```

---

## Expected Behavior (After Fix)

### Scenario 1: Same Choice
- **Question**: "Mummy or Work?"
- **AI**: "Mummy"
- **Human**: "Mummy"
- **Expected Score**: ~70-100 (high accuracy)
  - PREFERENCE: 100 (they agree)
  - EMOTIONAL: 50-80 (similar tone)
  - FACTUAL: 0 (no reasons given)
  - **Formula**: (100 × 0.7) + (50 × 0.3) + (0) = **85/100** ✅

### Scenario 2: Different Choice
- **Question**: "Mummy or Work?"
- **AI**: "Work"
- **Human**: "Mummy"
- **Expected Score**: ~0-23 (low accuracy)
  - PREFERENCE: 0 (they disagree)
  - EMOTIONAL: 50 (neutral)
  - FACTUAL: 0 (no reasons)
  - **Formula**: (0 × 0.7) + (50 × 0.3) + (0) = **15/100** ✅

---

## Scoring Formula

The judge uses **weighted scoring**:

### Standard Formula (when factual reasons exist):
```
Score = (Preference × 60%) + (Emotional × 30%) + (Factual × 10%)
```

### Adjusted Formula (when no reasons given):
```
Score = (Preference × 70%) + (Emotional × 30%)
```

---

## Files Changed
1. ✅ `engine/judge_handler.py`
   - Fixed intent_match derivation logic
   - Completely rewrote judge prompt with examples

---

## Testing
Test the fix by:
1. Asking a "this or that" question (e.g., "Mummy or Work?")
2. AI answers one option (e.g., "Work")
3. Human answers the **same** option -> should get high score (~70-100)
4. Human answers **different** option -> should get low score (~0-23)

---

## Impact
- ✅ Accurate scoring for binary choice questions
- ✅ No more false high scores for mismatched responses
- ✅ Intent badge correctly shows ✅/❌ based on preference score
- ✅ Scoring is now **logically consistent**
