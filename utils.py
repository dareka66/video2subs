def chunks_of_text(text: str, max_chars: int):
    buf = []
    size = 0
    for sent in text.split(". "):
        sent = sent.strip()
        if not sent:
            continue
        if size + len(sent) + 2 > max_chars and buf:
            yield ". ".join(buf) + "."
            buf = [sent]
            size = len(sent) + 1
        else:
            buf.append(sent)
            size += len(sent) + 2
    if buf:
        yield ". ".join(buf) + "."
