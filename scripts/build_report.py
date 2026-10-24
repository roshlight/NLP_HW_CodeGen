"""Build the PDF from recorded evidence; never invent official pass@1 or T4 timing."""

import argparse
import html
import json
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hw2-matplotlib")

ROOT = Path(__file__).resolve().parents[1]


def build(font):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

    summary = json.loads((ROOT / "artifacts/training_summary.json").read_text())
    if not summary.get("completed"):
        raise ValueError("Training must complete before producing the final report")
    metrics = [json.loads(s) for s in (ROOT / "artifacts/metrics.jsonl").read_text().splitlines()]
    validation = json.loads((ROOT / "artifacts/submission_validation.json").read_text())
    timing = json.loads((ROOT / "artifacts/submission.timing.json").read_text())
    revision_path = ROOT / "artifacts/revision_256/comparison.json"
    revision = json.loads(revision_path.read_text()) if revision_path.is_file() else None
    val = [r for r in metrics if "val/nll" in r]
    train = [r for r in metrics if "train/nll" in r]
    best = min(val, key=lambda r: r["val/nll"]) if val else None
    fig, ax = plt.subplots(figsize=(8, 3.7), layout="constrained")
    ax.plot([r["step"] for r in train], [r["train/nll"] for r in train], alpha=.6, linewidth=.8, label="Train NLL")
    ax.plot([r["step"] for r in val], [r["val/nll"] for r in val], linewidth=1.8, label="Validation NLL")
    ax.set(xlabel="Optimizer step", ylabel="Token NLL", yscale="log")
    ax.grid(alpha=.2)
    ax.legend()
    plot = ROOT / "report/training_curve.png"
    fig.savefig(plot, dpi=180)
    fig.savefig(ROOT / "report/training_curve.pdf")
    plt.close(fig)

    pdfmetrics.registerFont(TTFont("ReportFont", font))
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = "ReportFont"
    styles["Normal"].fontSize = 10
    styles["Normal"].leading = 15
    story = [Paragraph("HSE NLP HW2: генерация кода", styles["Title"]), Spacer(1, 14)]
    story.append(Paragraph("Первичная посылка: Transformer 106 977 792 параметра, контекст 256, генерация до 64 токенов.", styles["Normal"]))
    rows = [["Показатель", "Значение"], ["Шаги обучения", str(summary["steps"])],
            ["Обучение, часы", f"{summary['training_hours']:.3f}"],
            ["Лучший validation NLL", f"{best['val/nll']:.6f} на шаге {best['step']}" if best else "Нет замера"],
            ["Ответов / порядок", f"{validation['rows']} / {'проверен' if validation['order_verified'] else 'не проверен'}"],
            ["Пустых / синтаксически корректных", f"{validation['empty_outputs']} / {validation['syntax_valid']}"],
            ["GPU генерации", timing["device"]],
            ["Время инференса, включая загрузку", f"{timing['total_seconds']:.3f} с"],
            ["Достигнут EOS", f"{100 * timing['eos_rate']:.2f}%"],
            ["Public pass@1, исходные 64 токена", "47.71% (458/960), со слов пользователя" if revision else "Не измерен ботом"]]
    table = Table(rows, colWidths=[240, 255], hAlign="LEFT")
    table.setStyle(TableStyle([("FONTNAME", (0,0), (-1,-1), "ReportFont"),
        ("FONTSIZE", (0,0), (-1,-1), 9), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e5ecf5")),
        ("GRID", (0,0), (-1,-1), .3, colors.lightgrey), ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7)]))
    story += [Spacer(1, 12), table, Spacer(1, 12), Image(str(plot), width=495, height=229)]
    t4 = ROOT / "artifacts/t4.timing.json"
    if t4.is_file() and json.loads(t4.read_text()).get("actual_t4"):
        measured = json.loads(t4.read_text())
        statement = f"Фактический T4: {measured['total_seconds']:.3f} с. Лимит 180 с: {'выполнен' if measured['under_180_seconds'] else 'не выполнен'}."
    else:
        statement = "На реальной T4 замер отсутствует. Скорость H100 не доказывает соблюдение лимита 180 секунд."
    story += [Paragraph(statement, styles["Normal"]), PageBreak()]
    text = (ROOT / "report/experiments.md").read_text()
    for block in re.split(r"\n\s*\n", text.strip()):
        if block.startswith("#"):
            heading = re.sub(r"^#+\s*", "", block)
            story.append(Paragraph(html.escape(heading), styles["Heading2"]))
        else:
            story.append(Paragraph(html.escape(" ".join(block.splitlines())), styles["Normal"]))
            story.append(Spacer(1, 7))
    if revision:
        timing256 = json.loads((ROOT / "artifacts/revision_256/submission_256.timing.json").read_text())
        story += [PageBreak(), Paragraph("Исправление ограничения вывода: 64 → 256", styles["Heading1"])]
        comparison = [["Показатель (1919 вопросов)", "64 токена", "256 токенов"],
            ["Синтаксические ошибки", str(revision["max64"]["syntax_invalid"]), str(revision["max256"]["syntax_invalid"])],
            ["Без return среди разобранных", str(revision["max64"]["no_return_in_solve_among_parseable"]), str(revision["max256"]["no_return_in_solve_among_parseable"])],
            ["Завершились по EOS", str(revision["max64"]["eos_count"]), str(revision["max256"]["eos_count"])],
            ["Инференс на H100, с загрузкой", f"{timing['total_seconds']:.2f} с", f"{timing256['total_seconds']:.2f} с"],
            ["Public pass@1", "47.71% (458/960)", "Не измерен"]]
        comparison_table = Table(comparison, colWidths=[255, 120, 120], hAlign="LEFT")
        comparison_table.setStyle(TableStyle([("FONTNAME", (0,0), (-1,-1), "ReportFont"),
            ("FONTSIZE", (0,0), (-1,-1), 9), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e5ecf5")),
            ("GRID", (0,0), (-1,-1), .3, colors.lightgrey), ("TOPPADDING", (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7)]))
        story += [comparison_table, Spacer(1, 12), Paragraph(
            "Использованы те же веса шага 77000 и тот же токенизатор. Полный вывод был сохранён на H100 до получения результата бота. "
            "Подстановки эталонных ответов и переобучения на тестовых задачах не выполнялось. "
            "Проверки синтаксиса и наличия return не оценивают правильность решения. "
            "Смысловые ошибки модели сохраняются. Скорость на реальной T4 пока не измерена.", styles["Normal"])]
    target = ROOT / "report/report.pdf"
    SimpleDocTemplate(str(target), title="HSE NLP HW2 — code generation", author="HSE NLP HW2 submission").build(story)
    print(target)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--font", required=True, help="Path to a Cyrillic-capable TrueType font")
    args = p.parse_args()
    build(args.font)
