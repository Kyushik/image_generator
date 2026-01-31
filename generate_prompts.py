import json
import os
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a prompt builder for pixel art LoRA fine-tuning.

Task:
- I will provide a Korean keyword/subject (e.g., "하늘을 나는 고양이").
- You must output ONE English image-generation prompt.

Rules (must follow strictly):
1) Start the prompt with the subject translated into natural English.
2) Add a short, clear scene description (1–2 sentences max).
3) DO NOT add any extra lines, headers, bullet points, or explanations.
4) The final part of the prompt MUST be EXACTLY the following fixed suffix, appended at the very end with a comma before it.
5) Never change the words, order, spelling, punctuation, or spacing of the fixed suffix.
6) Ensure the output is a single line.

Fixed suffix (must be last, exact):
"large, clearly visible pixels, chunky pixel blocks, low resolution look, limited color palette, no smooth gradients, no anti-aliasing, no blur, sharp pixel edges, retro 16-bit game style"
"""

SUBJECT_GENERATOR_PROMPT = """You are a creative subject generator for pixel art images.

Generate ONE unique Korean subject/keyword for pixel art image generation.
The subject should be creative, visual, and interesting.

Rules:
1) Output ONLY the Korean subject, nothing else.
2) Keep it concise (3-10 words).
3) Make it visually interesting and suitable for pixel art.
4) Be creative and diverse - think of scenes, characters, objects, landscapes, etc.

Examples of good subjects:
- 하늘을 나는 고양이
- 석양을 바라보는 노부부
- 우주를 여행하는 로봇
- 마법의 숲속 오두막
- 비오는 날의 네온 도시

Generate a NEW, UNIQUE subject:"""


class PromptGenerator:
    def __init__(self, similarity_threshold: float = 0.85):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.history_file = Path("subjects_history.json")
        self.similarity_threshold = similarity_threshold
        self.history = self._load_history()

    def _load_history(self) -> dict:
        """Load history with subjects and their embeddings."""
        if self.history_file.exists():
            with open(self.history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"subjects": [], "embeddings": []}

    def _save_history(self):
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False)

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding using text-embedding-3-small."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _is_too_similar(self, embedding: list[float]) -> tuple[bool, float, str]:
        """Check if embedding is too similar to any existing subject."""
        if not self.history["embeddings"]:
            return False, 0.0, ""

        max_sim = 0.0
        most_similar = ""
        for i, existing_emb in enumerate(self.history["embeddings"]):
            sim = self._cosine_similarity(embedding, existing_emb)
            if sim > max_sim:
                max_sim = sim
                most_similar = self.history["subjects"][i]

        return max_sim >= self.similarity_threshold, max_sim, most_similar

    def generate_subject(self, max_retries: int = 5) -> str:
        """Generate a unique Korean subject using GPT with embedding similarity check."""
        for attempt in range(max_retries):
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": SUBJECT_GENERATOR_PROMPT}
                ],
                temperature=1.0,
                max_tokens=100
            )
            subject = response.choices[0].message.content.strip()

            # Get embedding and check similarity
            embedding = self._get_embedding(subject)
            is_similar, similarity, similar_to = self._is_too_similar(embedding)

            if not is_similar:
                # Save to history
                self.history["subjects"].append(subject)
                self.history["embeddings"].append(embedding)
                self._save_history()
                return subject
            else:
                print(f"  Retry {attempt+1}: '{subject}' too similar to '{similar_to}' ({similarity:.2f})")

        # If all retries failed, use the last one anyway
        print(f"  Warning: Using subject despite similarity after {max_retries} retries")
        self.history["subjects"].append(subject)
        self.history["embeddings"].append(embedding)
        self._save_history()
        return subject

    def generate_prompt(self, subject: str) -> str:
        """Generate an image prompt for the given subject."""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Now build the prompt for this subject:\n{subject}"}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    def generate(self, num_prompts: int = 1) -> list[dict]:
        """Generate multiple unique prompts."""
        results = []
        for i in range(num_prompts):
            print(f"[{i+1}/{num_prompts}] Generating subject...")
            subject = self.generate_subject()
            prompt = self.generate_prompt(subject)
            results.append({
                "subject": subject,
                "prompt": prompt
            })
            print(f"  Subject: {subject}")
        return results


def save_prompts(prompts: list[dict], output_file: str = "data/prompts.jsonl"):
    """Save prompts to JSONL file."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(prompts)} prompts to {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--output", type=str, default="data/prompts.jsonl")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold (0-1)")
    args = parser.parse_args()

    generator = PromptGenerator(similarity_threshold=args.threshold)
    prompts = generator.generate(args.num_prompts)
    save_prompts(prompts, args.output)
