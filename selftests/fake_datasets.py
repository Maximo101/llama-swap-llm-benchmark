"""In-memory stand-ins for the Hugging Face datasets, shaped like the real rows (no network needed)."""


class FakeDS(list):
    def select_columns(self, cols):
        return FakeDS([{c: r[c] for c in cols} for r in self])


def _qa(n, tag):
    letters = "ABCD"
    rows = []
    for i in range(n):
        gold = letters[i % 4]
        marker = " [[wrong]]" if i == 1 else " [[long]]" if i == 2 else ""
        rows.append((i, gold, marker))
    return rows


def gpqa(n=6):
    return FakeDS([{"question": f"Q{i}? A) a B) b C) c D) d [[ans:{g}]]{m}", "answer": g} for i, g, m in _qa(n, "g")])


def mmlu_pro(n=6):
    return FakeDS([{"question": f"Q{i}? [[ans:{g}]]{m}", "options": ["w", "x", "y", "z"], "answer": g,
                    "answer_index": "ABCD".index(g), "question_id": 100 + i, "category": "physics"}
                   for i, g, m in _qa(n, "m")])


def humaneval(n=4):
    rows = [{"task_id": "HE/0", "prompt": "def add(a, b):\n    \"\"\"add\"\"\"\n", "entry_point": "add",
             "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n"},
            {"task_id": "HE/1", "prompt": "def early(a):\n    \"\"\"identity\"\"\"\n", "entry_point": "early",
             "test": "def check(candidate):\n    assert candidate(1) == 2   # fails if actually run\n"}]
    return FakeDS(rows)


def mbpp(n=3):
    return FakeDS([{"task_id": 11 + i, "text": "Write a function to multiply two numbers (mbpp).",
                    "test_list": ["assert mul(2, 3) == 6", "assert mul(0, 5) == 0"], "test_setup_code": ""}
                   for i in range(n)])


def livecodebench():
    return FakeDS([{"question_id": "lcb1", "question_title": "Double", "question_content": "Print twice the number.",
                    "starter_code": "", "metadata": "{}",
                    "public_test_cases": '[{"input": "3\\n", "output": "6"}, {"input": "10\\n", "output": "20"}]',
                    "private_test_cases": ""}])


def gsm8k(n=6):
    rows = []
    for i in range(n):
        gold = str(10 * (i + 1))
        marker = " [[wrong]]" if i == 1 else " [[long]]" if i == 2 else ""
        rows.append({"question": f"Tom has {i} apples ... [[ans:{gold}]]{marker}",
                     "answer": f"step one\nstep two\n#### {gold if i != 0 else '1,0'}".replace("1,0", "10")})
    return FakeDS(rows)


def math500(n=6):
    golds = ["\\frac{1}{2}", "3", "(1, 2)", "12", "\\dfrac{3}{4}", "x=5"]
    answers = ["0.5", "3", "(1,2)", "12 cm", "\\frac{3}{4}", "5"]       # model answer vs gold: all equivalent
    return FakeDS([{"problem": f"Compute #{i}. [[ans:{answers[i]}]]", "answer": golds[i], "unique_id": f"test/{i}.json",
                    "subject": "Algebra", "level": 2, "solution": "..."} for i in range(n)])


BY_KEY = {"gpqa": gpqa, "mmlu_pro": mmlu_pro, "humaneval": humaneval, "mbpp": mbpp, "livecodebench": livecodebench,
          "gsm8k": gsm8k, "math": math500}
