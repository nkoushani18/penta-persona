"""
LLM Judge Handler
Interfaces with Ollama LLaMA 3.2 to judge semantic similarity between AI and human responses.
"""

import requests
import json
import os
import google.generativeai as genai
from typing import Dict, Optional


class JudgeHandler:
    """Handler for LLM-based response judging using Ollama."""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """
        Initialize the judge handler.
        
        Args:
            ollama_url: URL of the Ollama server (default: localhost:11434)
        """
        self.ollama_url = ollama_url
        self.model = "llama3.2"  # LLaMA 3.2 model
        self.api_keys = []
        for i in range(1, 5):
            key = os.environ.get(f"GEMINI_API_KEY_{i}")
            if key:
                self.api_keys.append(key)
        self.current_key_index = 0
        # Once Ollama fails once, skip it for the rest of the session (ultra-fast fallback)
        self._ollama_available = None  # None = not yet checked
        
    def check_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def judge_responses(
        self, 
        user_question: str, 
        ai_response: str, 
        human_response: str,
        persona_name: str
    ) -> Dict:
        """
        Judge the semantic similarity between AI and human responses.
        
        Args:
            user_question: The original user question
            ai_response: The AI agent's response
            human_response: The human's response
            persona_name: Name of the persona being evaluated
            
        Returns:
            Dict with:
                - score: 0-100 semantic accuracy score
                - breakdown: Detailed analysis
                - intent_match: Whether core intents match
                - reasoning: Explanation of the score
        """
        # Build the prompt first (needed for both Ollama and Gemini fallback)
        prompt = self._create_judge_prompt(
            user_question, ai_response, human_response, persona_name
        )

        # Fast-path: if Ollama already failed once this session, skip directly to Gemini
        if self._ollama_available is False:
            print("[JUDGE] Ollama previously unavailable. Going straight to Gemini...", flush=True)
            return self._fallback_to_gemini(prompt)

        # Check if Ollama is running
        print("[JUDGE] Checking Ollama connection...", flush=True)
        if not self.check_connection():
            print("[JUDGE] Ollama not detected. Switching to Gemini Cloud Judge instantly...", flush=True)
            self._ollama_available = False  # Never check again this session
            return self._fallback_to_gemini(prompt)

        self._ollama_available = True
        
        try:
            print(f"[JUDGE] Sending to Ollama LLaMA 3.2... (this may take 10-30 seconds)", flush=True)
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more consistent judging
                        "top_p": 0.9,
                        "num_predict": 300,  # Limit response length for faster results
                    }
                },
                timeout=10  # 10s max — if slow, fall back to Gemini immediately
            )
            
            print(f"[JUDGE] Got response from Ollama (status: {response.status_code})", flush=True)
            
            if response.status_code != 200:
                print(f"[JUDGE] Ollama API error: {response.status_code}. Falling back to Gemini...", flush=True)
                return self._fallback_to_gemini(prompt)
            
            result = response.json()
            judge_output = result.get("response", "")
            
            print(f"[JUDGE] Parsing judge output...", flush=True)
            # Parse the judge's output
            parsed = self._parse_judge_output(judge_output)
            
            print(f"[JUDGE] ✅ Judging complete! Score: {parsed.get('score', 0)}/100", flush=True)
            return parsed
            
        except requests.exceptions.Timeout:
            print("[JUDGE] ⏱️ Ollama timed out (>10s). Switching to Gemini instantly...", flush=True)
            self._ollama_available = False
            return self._fallback_to_gemini(prompt)
        except Exception as e:
            print(f"[JUDGE] ❌ Ollama Error: {str(e)}. Switching to Gemini instantly...", flush=True)
            self._ollama_available = False
            return self._fallback_to_gemini(prompt)

    def _fallback_to_gemini(self, prompt: str) -> Dict:
        """Fallback to Gemini if Ollama is not available."""
        if not self.api_keys:
            # Try to use hardcoded key if no env keys are found
            self.api_keys.append("AIzaSyAg3zxRqSsYv8W17gwQkmZs-LhL1AOEQUc")
            
        max_retries = len(self.api_keys)
        for attempt in range(max_retries):
            try:
                key = self.api_keys[self.current_key_index]
                # Re-configure to prevent global state conflicts
                genai.configure(api_key=key)
                # Use gemini-2.5-flash as the fast judge
                model = genai.GenerativeModel('models/gemini-2.5-flash-lite')
                
                print(f"[JUDGE] Sending to Gemini (Key {self.current_key_index + 1})...", flush=True)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.3,
                        top_p=0.9,
                        max_output_tokens=300,
                    )
                )
                
                if response.text:
                    judge_output = response.text.strip()
                    parsed = self._parse_judge_output(judge_output)
                    print(f"[JUDGE] ✅ Gemini Judging complete! Score: {parsed.get('score', 0)}/100", flush=True)
                    return parsed
                    
            except Exception as e:
                error_str = str(e)
                print(f"[JUDGE] Gemini fallback error: {error_str}", flush=True)
                if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower() or "403" in error_str:
                    print(f"[JUDGE] Rotating Gemini key...", flush=True)
                    self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                else:
                    # Break on non-quota errors
                    break
                    
        return {
            "error": "All Gemini fallback keys exhausted or failed.",
            "score": 0,
            "breakdown": {},
            "reasoning": "Failed to evaluate using cloud fallback"
        }
    
    def _create_judge_prompt(
        self, 
        question: str, 
        ai_resp: str, 
        human_resp: str,
        persona_name: str
    ) -> str:
        """Ultra-strict judge prompt with explicit examples."""
        prompt = f"""You are a strict judge evaluating if AI and Human gave the SAME answer.

QUESTION: "{question}"
AI ANSWER: "{ai_resp}"
HUMAN ANSWER: "{human_resp}"

=== COMPARISON TABLE ===
| Example                  | AI Says | Human Says | PREFERENCE | INTENT_MATCH |
|--------------------------|---------|------------|------------|--------------|
| Same choice              | "Work"  | "Work"     | 100        | YES          |
| Different choice         | "Work"  | "Mummy"    | 0          | NO           |
| Same word, same meaning  | "Lion"  | "Lion"     | 100        | YES          |
| Different words          | "Lion"  | "Tiger"    | 0          | NO           |

=== YOUR TASK ===
Compare the AI and Human answers above:
1. Do they say the EXACT SAME THING (or same meaning)?
   - If YES -> PREFERENCE: 100, INTENT_MATCH: YES
   - If NO -> PREFERENCE: 0, INTENT_MATCH: NO

2. EMOTIONAL score:
   - 80 if tone similar
   - 50 if neutral/can't tell
   - 0 if completely opposite tone

3. FACTUAL score:
   - 0 if no reasons given (just 1-word answers)
   - 0 if different reasons
   - 80 if similar reasons

=== OUTPUT FORMAT ===
PREFERENCE: [0 or 100]
EMOTIONAL: [0-100]
FACTUAL: [0-100]
INTENT_MATCH: [YES or NO]
REASONING: [One sentence explaining why they match or don't match]"""
        
        return prompt
    
    def _parse_judge_output(self, output: str) -> Dict:
        """Parse strict judge output format."""
        # Clean output
        output = output.strip()
        print(f"[JUDGE DEBUG] Raw LLM output:\n{output}\n", flush=True)

        result = {
            "score": 0,
            "intent_match": False,
            "breakdown": {
                "preference_alignment": 0,
                "emotional_alignment": 0,
                "factual_alignment": 0
            },
            "reasoning": "",
            "raw_output": output
        }

        # Parse line by line
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            upper_line = line.upper()

            # Parse Scores
            if "PREFERENCE:" in upper_line:
                try:
                    score_part = line.split(':', 1)[1]
                    score = int(''.join(filter(str.isdigit, score_part)))
                    result["breakdown"]["preference_alignment"] = score
                except: pass
            
            elif "EMOTIONAL:" in upper_line:
                try:
                    score_part = line.split(':', 1)[1]
                    score = int(''.join(filter(str.isdigit, score_part)))
                    result["breakdown"]["emotional_alignment"] = score
                except: pass

            elif "FACTUAL:" in upper_line:
                try:
                    score_part = line.split(':', 1)[1]
                    score = int(''.join(filter(str.isdigit, score_part)))
                    result["breakdown"]["factual_alignment"] = score
                except: pass
            
            # Parse Intent
            elif "INTENT_MATCH:" in upper_line:
                if "YES" in upper_line:
                    result["intent_match"] = True

            # Parse Reasoning
            elif "REASONING:" in upper_line:
                try:
                    result["reasoning"] = line.split(':', 1)[1].strip()
                except:
                    result["reasoning"] = line

        # LOGIC FIX: Derive intent_match FROM preference score (not vice versa)
        # For "this or that" questions, if they chose different options, preference should be 0
        # Intent matches only if preference alignment is reasonably high
        p = result["breakdown"]["preference_alignment"]
        if p >= 70:
            result["intent_match"] = True
            print(f"[JUDGE] ✅ Intent Match: Preference score is {p} (>= 70)", flush=True)
        else:
            result["intent_match"] = False
            print(f"[JUDGE] ❌ Intent Mismatch: Preference score is {p} (< 70)", flush=True)

        # Calculate Final Score using Formula
        p = result["breakdown"]["preference_alignment"]
        e = result["breakdown"]["emotional_alignment"]
        f = result["breakdown"]["factual_alignment"]
        
        if f == 0:
            # If Factual is 0 (N/A), redistribute weight: Intent 70%, Tone 30%
            final_score = int((p * 0.7) + (e * 0.3))
            print(f"[JUDGE] Factual is 0 -> Redistributed weights: P:{p}*0.7 + E:{e}*0.3 = {final_score}", flush=True)
        else:
            final_score = int((p * 0.6) + (e * 0.3) + (f * 0.1))
            print(f"[JUDGE] Standard weights -> P:{p} E:{e} F:{f} = {final_score}", flush=True)
            
        result["score"] = final_score

        return result
