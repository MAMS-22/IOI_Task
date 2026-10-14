"""Aligned, answer-free IOI prefixes and explicit semantic token positions."""
import random
import re
from collections import defaultdict

# Wang et al.'s 15 short templates, from the vendored Easy-Transformer
# easy_transformer/ioi_dataset.py, BABA_TEMPLATES. The multilingual paper
# does not supply its exact lexical pools or sampling conventions.
TEMPLATE_SOURCE = "redwoodresearch/Easy-Transformer: easy_transformer/ioi_dataset.py BABA_TEMPLATES"
BABA_TEMPLATES = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]
NAMES = "John Mary Michael Jessica David Sarah Robert Lisa William Jennifer James Emily Daniel Ashley Matthew Amanda Andrew Laura Ryan Rebecca Brian Michelle Kevin Rachel Thomas Nicole Steven Anna Joseph Emma Susan Linda Patricia Elizabeth Barbara Charles Anthony Mark Paul George".split()
PLACES = "store market office garden school restaurant library station park beach".split()
OBJECTS = "apple book drink flower gift letter meal snack toy ticket".split()
ROLES = ("IO", "S1", "S1+1", "S2", "END")


def name_pool(tokenizer, candidates=None):
    names = [n for n in (NAMES if candidates is None else candidates)
             if len(tokenizer.encode(" " + n, add_special_tokens=False)) == 1]
    if len(names) < 5:
        raise ValueError("Need at least five single-token names for disjoint ABC controls.")
    return names


def name_positions(text, name, offsets):
    positions = []
    for match in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text):
        indices = [i for i, (a, b) in enumerate(offsets)
                   if b > a and a < match.end() and b > match.start()]
        if len(indices) != 1:
            raise ValueError(f"Name {name!r} is not a single token in context: {text!r}")
        positions.append(indices[0])
    return positions


def abc_control(example, tokenizer, rng, names):
    """Replace the three prefix name tokens by disjoint C,D,E, with no repetition."""
    c, d, e = rng.sample([n for n in names if n not in (example['A'], example['B'])], 3)
    ids = example["clean_ids"].copy()
    for role, name in zip(("IO", "S1", "S2"), (c, d, e)):
        ids[example["positions"][role]] = tokenizer.encode(" " + name, add_special_tokens=False)[0]
    return ids


def build_examples(tokenizer, n_per_pattern=200, seed=42, corruption="abc", prepend_bos=True,
                   exclude_prompts=(), vocabulary=None):
    if n_per_pattern < 1 or corruption not in ("abc", "subject_swap"):
        raise ValueError("Positive sample count and abc/subject_swap corruption required.")
    rng = random.Random(seed)
    names = name_pool(tokenizer, None if vocabulary is None else vocabulary['names'])
    places = PLACES if vocabulary is None else vocabulary['places']
    objects = OBJECTS if vocabulary is None else vocabulary['objects']
    if not places or not objects:
        raise ValueError('Nonempty place and object pools required')
    bos = tokenizer.bos_token_id
    if prepend_bos and bos is None:
        raise ValueError("Explicit BOS requested but tokenizer has no BOS token.")
    shift = int(prepend_bos)
    examples = []
    seen = set(exclude_prompts)
    # Paired patterns share names, object, place and template. Each sample index
    # cycles through the template bank so counts differ by at most one.
    for i in range(n_per_pattern):
        for attempt in range(1000):
            a, b = rng.sample(names, 2)
            place, obj = rng.choice(places), rng.choice(objects)
            pair = []
            for pattern in ("ABBA", "BABA"):
                template = BABA_TEMPLATES[i % 15]
                if pattern == "ABBA":
                    template = template.replace("[B] and [A]", "[A] and [B]", 1)
                prefix_template = template.removesuffix(" [A]")
                prompt = (prefix_template.replace("[A]", a).replace("[B]", b)
                          .replace("[PLACE]", place).replace("[OBJECT]", obj))
                if prompt in seen:
                    break
                encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
                ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
                io_pos = name_positions(prompt, a, offsets)
                s_pos = name_positions(prompt, b, offsets)
                if len(io_pos) != 1 or len(s_pos) != 2:
                    raise ValueError("Expected one IO and two subject mentions in prefix.")
                a_id = tokenizer.encode(" " + a, add_special_tokens=False)[0]
                b_id = tokenizer.encode(" " + b, add_special_tokens=False)[0]
                if tokenizer.encode(prompt + " " + a, add_special_tokens=False) != ids + [a_id]:
                    raise ValueError("Prefix/answer token boundary is not stable.")
                positions = dict(IO=io_pos[0] + shift, S1=s_pos[0] + shift,
                                 S2=s_pos[1] + shift, END=len(ids) - 1 + shift)
                positions["S1+1"] = positions["S1"] + 1
                clean_ids = ([bos] if prepend_bos else []) + ids
                if any(clean_ids[p] != token for p, token in (
                        (positions["IO"], a_id), (positions["S1"], b_id), (positions["S2"], b_id))):
                    raise ValueError("Name token identity differs across context positions.")
                ex = dict(example_id=f"{i:05d}_{pattern}", pattern=pattern, template_id=i % 15 + 1,
                          prompt=prompt, A=a, B=b, place=place, object=obj,
                          clean_ids=clean_ids, positions=positions, io_id=a_id, s_id=b_id)
                if corruption == "abc":
                    corrupt_ids = abc_control(ex, tokenizer, rng, names)
                else:
                    corrupt_ids = clean_ids.copy()
                    corrupt_ids[positions["S2"]] = a_id
                ex.update(corrupt_ids=corrupt_ids, corruption=corruption,
                          corrupt_prompt=tokenizer.decode(corrupt_ids[shift:]))
                changed = [j for j, (x, y) in enumerate(zip(clean_ids, corrupt_ids)) if x != y]
                expected = sorted(positions[r] for r in (("IO", "S1", "S2") if corruption == "abc" else ("S2",)))
                if changed != expected:
                    raise ValueError("Corruption did not preserve exact token alignment.")
                pair.append(ex)
            if len(pair) == 2:
                examples.extend(pair)
                seen.update(ex["prompt"] for ex in pair)
                break
        else:
            raise RuntimeError("Unable to generate unique examples.")
    return examples


def batches(examples, batch_size):
    """Equal-length groups avoid padding, masking and ALiBi alignment ambiguities."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    groups = defaultdict(list)
    for ex in examples:
        groups[len(ex["clean_ids"])].append(ex)
    for length in sorted(groups):
        group = groups[length]
        for i in range(0, len(group), batch_size):
            yield group[i:i + batch_size]
