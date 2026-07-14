"""Pinned IFBench loose verifier for the selected regression-task instructions.

The checker behavior is transcribed from allenai/IFBench commit
1091c4c3de6c1f6ed12c012ed68f11ea450b0117 (Apache-2.0).  Only instruction
types present in the frozen task data are implemented.
"""

from collections import Counter
import re
import string


def configure_nltk_data(path):
    """Restrict NLTK to the authenticated task-local resource tree."""
    import nltk

    value = str(path)
    previous = tuple(nltk.data.path)
    nltk.data.path[:] = [value]
    cached_punkt = getattr(nltk.tokenize, "_get_punkt_tokenizer", None)
    if cached_punkt is not None and hasattr(cached_punkt, "cache_clear"):
        cached_punkt.cache_clear()
    return previous


def _sentences(value):
    import nltk

    return nltk.sent_tokenize(value)


def _words(value):
    import nltk

    return nltk.tokenize.RegexpTokenizer(r"\w+").tokenize(value)


def _check(instruction_id, kwargs, value):
    if instruction_id == "count:word_count_range":
        count = len(_words(value))
        return kwargs["min_words"] <= count <= kwargs["max_words"]
    if instruction_id == "ratio:stop_words":
        import nltk

        words = _words(value)
        if not words:
            return False
        stop = set(nltk.corpus.stopwords.words("english"))
        percentage = 100 * sum(word.lower() in stop for word in words) / len(words)
        return percentage <= kwargs["percentage"]
    if instruction_id == "words:alphabet":
        cleaned = value.translate(str.maketrans("", "", string.punctuation))
        words = cleaned.strip(string.punctuation + " ").split()
        if not words:
            return False
        alphabet = string.ascii_lowercase
        letter = words[0][0].lower()
        if letter not in alphabet:
            return False
        for word in words[1:]:
            word = word.strip(string.punctuation + " ").lower()
            if not word:
                continue
            letter = alphabet[(alphabet.index(letter) + 1) % 26]
            if word[0] != letter:
                return False
        return True
    if instruction_id == "sentence:alliteration_increment":
        previous = -1
        for sentence in _sentences(value):
            words = [
                word.lstrip(string.punctuation + " ")
                for word in sentence.lower().split()
            ]
            words = [word for word in words if word]
            count, continuing = 0, False
            for left, right in zip(words, words[1:]):
                if left[0] == right[0]:
                    count += 1 if continuing else 2
                    continuing = True
                else:
                    continuing = False
            if count <= previous:
                return False
            previous = count
        return True
    if instruction_id == "words:prime_lengths":
        cleaned = value.translate(str.maketrans("", "", string.punctuation))
        primes = {
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            29,
            31,
            37,
            41,
            43,
            47,
            53,
            59,
            61,
            67,
            71,
            73,
            79,
            83,
            89,
            97,
        }
        return all(len(word) in primes for word in cleaned.split())
    if instruction_id == "format:options":
        options = kwargs["options"]
        strict = re.match(r"\W*[aA]\W*[bB]\W*[cC]\W*", options) is not None
        separator = "/" if "/" in options else ("or" if "or" in options else ",")
        choices = [item.strip() for item in options.split(separator)]
        if strict:
            return value in choices
        normalized = value.strip(string.punctuation + " ").lower()
        return any(
            choice.strip(string.punctuation + " ").lower() == normalized
            for choice in choices
        )
    if instruction_id == "format:newline":
        cleaned = value.translate(str.maketrans("", "", string.punctuation))
        lines = [line for line in cleaned.strip().split("\n") if line]
        return len(lines) == len(cleaned.strip().split())
    if instruction_id == "format:emoji":
        import emoji

        sentences = _sentences(value)
        for index, sentence in enumerate(sentences):
            stripped = sentence.translate(
                str.maketrans("", "", string.punctuation)
            ).strip()
            if not stripped:
                return False
            last = stripped[-1]
            second_last = stripped[-2] if len(stripped) > 1 else last
            if not emoji.is_emoji(last) and not emoji.is_emoji(second_last):
                if index >= len(sentences) - 1:
                    return False
                following = (
                    sentences[index + 1]
                    .translate(str.maketrans("", "", string.punctuation))
                    .strip()
                )
                if not following or not emoji.is_emoji(following[0]):
                    return False
        return True
    if instruction_id == "words:start_verb":
        import nltk

        tokens = nltk.word_tokenize(value)
        tagged = nltk.pos_tag(tokens)
        return bool(tagged) and "VB" in tagged[0][1]
    if instruction_id == "words:repeats":
        words = (
            value.lower().translate(str.maketrans("", "", string.punctuation)).split()
        )
        return all(count <= kwargs["small_n"] for count in Counter(words).values())
    if instruction_id == "words:last_first":
        sentences = _sentences(value)
        for left, right in zip(sentences, sentences[1:]):
            last = left.rstrip(string.punctuation + " ").split()
            first = right.lstrip(string.punctuation + " ").split()
            if not last or not first or last[-1].lower() != first[0].lower():
                return False
        return True
    if instruction_id == "sentence:increment":
        sentences = _sentences(value)
        if not sentences:
            return False
        counts = [
            len(
                sentence.translate(str.maketrans("", "", string.punctuation))
                .strip()
                .split()
            )
            for sentence in sentences
        ]
        return all(
            right == left + kwargs["small_n"] for left, right in zip(counts, counts[1:])
        )
    if instruction_id == "format:line_indent":
        lines = [line for line in value.split("\n") if line.strip()]
        indents = [len(line) - len(line.lstrip(" ")) for line in lines]
        return all(right > left for left, right in zip(indents, indents[1:]))
    if instruction_id == "format:quote_unquote":
        compact = "".join(
            value.replace("“", '"').replace("”", '"').replace("'\"'", "").split()
        )
        if '""' in compact:
            return False
        stripped = compact.strip(string.digits + string.punctuation.replace('"', ""))
        return not stripped or stripped[-1] != '"'
    if instruction_id == "format:thesis":
        index = value.find("<i>")
        tag = "i"
        if index == -1:
            index, tag = value.find("<em>"), "em"
        if index == -1:
            return False
        local = value[index:]
        end = local.find(f"</{tag}>")
        if end == -1 or not local[len(tag) + 2 : end].strip():
            return False
        return bool(local[end + len(tag) + 3 :].strip())
    if instruction_id == "format:sub-bullets":
        return all("-" in bullet for bullet in value.split("*")[1:])
    if instruction_id == "format:title_case":
        import nltk

        for word in nltk.word_tokenize(value):
            if not word or not word[0].isalpha():
                continue
            if len(word) == 1:
                if word[0].islower():
                    return False
            elif word[0].islower() and word[1:].isupper():
                return False
            elif word[0].islower() and word[1:].islower():
                return False
        return True
    if instruction_id == "format:no_whitespace":
        return not any(character.isspace() for character in value)
    raise ValueError(f"unsupported frozen IFBench instruction: {instruction_id}")


def loose_pass(row, response):
    """Match IFBench's eight loose response transformations."""
    if response is None:
        return False
    lines = response.split("\n")
    variants = [
        response,
        "\n".join(lines[1:]).strip(),
        "\n".join(lines[:-1]).strip(),
        "\n".join(lines[1:-1]).strip(),
    ]
    variants += [value.replace("*", "") for value in variants]
    for instruction_id, raw_kwargs in zip(row["instruction_id_list"], row["kwargs"]):
        kwargs = {key: value for key, value in raw_kwargs.items() if value is not None}
        if not any(
            value.strip() and _check(instruction_id, kwargs, value)
            for value in variants
        ):
            return False
    return True
