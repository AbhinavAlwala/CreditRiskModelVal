"""Generate the concise model validation report in reports/."""
from pathlib import Path
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parent
RESULTS = json.loads((ROOT / "outputs" / "results.json").read_text())
REPORT = ROOT / "reports" / "model_validation_report.pdf"

p = RESULTS["primary"]
c = RESULTS["challenger"]
ci = RESULTS["primary_confidence_intervals"]
auc_delta = RESULTS["auc_difference_primary_minus_challenger"]
stress = {r["scenario"]: r for r in RESULTS["stress"]}
backtest = RESULTS["annual_backtest"]

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20, leading=23, spaceAfter=4, textColor=colors.HexColor("#111111")))
styles.add(ParagraphStyle(name="Kicker", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8.2, leading=10, textColor=colors.HexColor("#555555"), spaceAfter=8))
styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, spaceBefore=7, spaceAfter=5, textColor=colors.HexColor("#111111")))
styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.8, leading=12.2, spaceAfter=5, textColor=colors.HexColor("#222222")))
styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.2, leading=9.3, textColor=colors.HexColor("#555555")))
styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=8.8, leading=12, borderPadding=7, backColor=colors.HexColor("#F2F2F2"), textColor=colors.HexColor("#111111")))


def header_footer(canvas, doc):
    canvas.saveState()
    w, _ = letter
    canvas.setStrokeColor(colors.HexColor("#D8D8D8"))
    canvas.setLineWidth(0.5)
    canvas.line(0.55 * inch, 0.48 * inch, w - 0.55 * inch, 0.48 * inch)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(0.55 * inch, 0.29 * inch, "Credit Risk Model Validation - synthetic demonstration portfolio")
    canvas.drawRightString(w - 0.55 * inch, 0.29 * inch, f"Page {doc.page}")
    canvas.restoreState()


def metric_card(label, value, note=""):
    data = [
        [Paragraph(f"<b>{value}</b>", ParagraphStyle("mv", fontName="Helvetica-Bold", fontSize=15.5, leading=18, textColor=colors.HexColor("#111111")))],
        [Paragraph(label, ParagraphStyle("ml", fontName="Helvetica-Bold", fontSize=7.1, leading=8.5, textColor=colors.HexColor("#555555")))],
        [Paragraph(note, ParagraphStyle("mn", fontName="Helvetica", fontSize=6.4, leading=7.7, textColor=colors.HexColor("#777777")))],
    ]
    t = Table(data, colWidths=[1.48 * inch], rowHeights=[0.26 * inch, 0.16 * inch, 0.22 * inch])
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CFCFCF")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def simple_table(data, widths, font=7.4):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAEAEA")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("LEADING", (0, 0), (-1, -1), font + 1.5),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0D0D0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


story = [
    Paragraph("CREDIT RISK / MODEL RISK", styles["Kicker"]),
    Paragraph("Model Validation & Loss Sensitivity", styles["Title2"]),
    Paragraph(
        "A reproducible demonstration of PD development, out-of-time validation, statistical uncertainty, challenger benchmarking, temporal backtesting, and expected credit loss sensitivity.",
        styles["Bodyx"],
    ),
    Spacer(1, 3),
]

auc_ci = ci["roc_auc"]
cards = Table([[
    metric_card("ROC-AUC", f"{p['roc_auc']:.3f}", "2025 OOT"),
    metric_card("AUC 95% CI", f"{auc_ci['ci_lower']:.3f}-{auc_ci['ci_upper']:.3f}", "500 bootstraps"),
    metric_card("KS", f"{p['ks_statistic']:.3f}", "Discrimination"),
    metric_card("Brier", f"{p['brier_score']:.3f}", "Probability error"),
]], colWidths=[1.55 * inch] * 4)
cards.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
story += [cards, Spacer(1, 8)]

story += [Paragraph("1. Validation design", styles["H2x"])]
story += [Paragraph(
    f"The model estimates 12-month probability of default using seven borrower-level variables. Development uses {RESULTS['development_rows']:,} observations from 2022-2024; the 2025 vintage ({RESULTS['oot_validation_rows']:,} observations) is held out for final validation. Automated data controls pass {RESULTS['data_quality_checks_passed']}/{RESULTS['data_quality_checks_total']} checks before model fitting.",
    styles["Bodyx"],
)]

story += [Paragraph("2. Challenger and uncertainty", styles["H2x"])]
story += [Paragraph(
    f"The logistic primary model achieves AUC {p['roc_auc']:.3f}, compared with {c['roc_auc']:.3f} for the shallow decision-tree challenger. The paired-bootstrap AUC difference is +{auc_delta['auc_difference']:.3f}, with a 95% interval of +{auc_delta['ci_lower']:.3f} to +{auc_delta['ci_upper']:.3f}. This quantifies uncertainty in the model comparison rather than relying on two point estimates.",
    styles["Callout"],
)]

bt_data = [["Vintage", "AUC", "KS", "Brier", "Predicted PD", "Observed DR"]]
for r in backtest:
    bt_data.append([
        str(int(r["validation_year"])), f"{r['roc_auc']:.3f}", f"{r['ks_statistic']:.3f}", f"{r['brier_score']:.3f}",
        f"{r['mean_predicted_pd']:.2%}", f"{r['observed_default_rate']:.2%}",
    ])
story += [Paragraph("3. Rolling out-of-time backtest", styles["H2x"]), simple_table(bt_data, [0.68*inch, 0.65*inch, 0.58*inch, 0.63*inch, 1.12*inch, 1.08*inch])]

backtest_img = Image(str(ROOT / "outputs" / "figures" / "annual_backtest.png"), width=4.35 * inch, height=3.02 * inch)
backtest_img.hAlign = "CENTER"
story += [Spacer(1, 6), backtest_img]

cal_gap = p["observed_default_rate"] - p["mean_predicted_pd"]
story += [Paragraph(
    f"Monitoring signal: 2025 mean predicted PD is {p['mean_predicted_pd']:.2%} versus a {p['observed_default_rate']:.2%} observed default rate, a {cal_gap:.2%} absolute shortfall. Discrimination remains useful, but calibration has deteriorated.",
    styles["Smallx"],
)]

story += [PageBreak(), Paragraph("STRESS / BUSINESS TRANSLATION", styles["Kicker"]), Paragraph("Expected Credit Loss Sensitivity", styles["Title2"])]
story += [Paragraph(
    "The synthetic portfolio includes exposure at default (EAD) and loss given default (LGD) fields that are excluded from PD model inputs. They are used only after scoring to translate PD into financial loss using ECL = PD x EAD x LGD.",
    styles["Bodyx"],
)]

stress_data = [["Scenario", "Mean PD", "Mean LGD", "Expected loss", "Change vs base"]]
for name in ["baseline", "mild", "severe"]:
    r = stress[name]
    stress_data.append([
        name.title(), f"{r['mean_pd']:.2%}", f"{r['mean_lgd']:.2%}", f"${r['expected_loss']/1_000_000:.2f}M",
        "-" if name == "baseline" else f"{r['loss_change_vs_baseline']:+.1%}",
    ])
story += [simple_table(stress_data, [1.0*inch, 0.95*inch, 0.95*inch, 1.3*inch, 1.2*inch], font=8.0), Spacer(1, 7)]

loss_img = Image(str(ROOT / "outputs" / "figures" / "expected_loss.png"), width=4.55 * inch, height=3.15 * inch)
loss_img.hAlign = "CENTER"
story += [loss_img]

story += [Paragraph("4. Validation conclusion", styles["H2x"])]
story += [Paragraph(
    f"The logistic model is retained as the preferred benchmark because it materially outperforms the challenger and maintains stable discrimination across three rolling validation vintages. It is not production-ready: 2025 calibration underpredicts realized default, and baseline expected loss (${stress['baseline']['expected_loss']/1_000_000:.2f}M) is below the synthetic realized loss (${RESULTS['realized_oot_loss']/1_000_000:.2f}M). A production decision would require recalibration, additional history, independent validation and broader governance testing.",
    styles["Callout"],
)]

story += [Paragraph("5. Limitations", styles["H2x"])]
for text in [
    "Synthetic data: no real borrowers or proprietary bank data are used.",
    "Illustrative EAD/LGD and stress assumptions: values are not calibrated to a real portfolio or supervisory path.",
    "Limited history: three backtest vintages demonstrate method, not production sufficiency.",
    "No causal or fair-lending conclusion: coefficients are predictive associations and the project is not a regulatory review.",
]:
    story.append(Paragraph("- " + text, styles["Bodyx"]))

story += [Spacer(1, 2), Paragraph(
    "Decision-use note: educational portfolio project only. Not a lending policy, regulatory validation, financial advice, or representation of any bank's production models.",
    styles["Smallx"],
)]

REPORT.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(str(REPORT), pagesize=letter, rightMargin=0.55*inch, leftMargin=0.55*inch, topMargin=0.48*inch, bottomMargin=0.58*inch)
doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
print(REPORT)
