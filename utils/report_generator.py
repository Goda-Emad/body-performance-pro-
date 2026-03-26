"""
Report Generator Module
-----------------------
Generates PDF reports and summaries for the Body Performance Analytics application.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO


class ReportGenerator:
    """
    Generate PDF reports for model predictions and analysis.
    """
    
    def __init__(self, title: str = "Body Performance Analytics Report"):
        """
        Initialize report generator.
        
        Parameters:
        -----------
        title : str
            Report title
        """
        self.title = title
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
        
    def _create_custom_styles(self):
        """Create custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            alignment=TA_CENTER,
            spaceAfter=30,
            textColor=colors.HexColor('#1e3a5f')
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#2c4e6e')
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomSubheading',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceAfter=8,
            textColor=colors.HexColor('#64748b')
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            leading=14
        ))
        
        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=18,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1e3a5f'),
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='MetricLabel',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#64748b')
        ))
        
    def _create_table(self, data: pd.DataFrame, title: str = None) -> List:
        """
        Create a formatted table from dataframe.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to display
        title : str, optional
            Table title
        
        Returns:
        --------
        List
            List of reportlab elements
        """
        elements = []
        
        if title:
            elements.append(Paragraph(title, self.styles['CustomHeading']))
            elements.append(Spacer(1, 6))
        
        # Convert dataframe to list of lists
        table_data = [data.columns.tolist()] + data.values.tolist()
        
        # Create table
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 1), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _create_metric_card(self, value: str, label: str, icon: str = "") -> List:
        """
        Create a metric card element.
        
        Parameters:
        -----------
        value : str
            Metric value
        label : str
            Metric label
        icon : str
            Icon emoji
        
        Returns:
        --------
        List
            List of reportlab elements
        """
        elements = []
        
        if icon:
            elements.append(Paragraph(f"{icon}", self.styles['MetricValue']))
        elements.append(Paragraph(value, self.styles['MetricValue']))
        elements.append(Paragraph(label, self.styles['MetricLabel']))
        
        return elements
    
    def _fig_to_image(self, fig) -> BytesIO:
        """
        Convert matplotlib figure to BytesIO object for reportlab.
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Figure to convert
        
        Returns:
        --------
        BytesIO
            Image buffer
        """
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf
    
    def generate_report(
        self,
        input_data: Dict[str, Any],
        predictions: Dict[str, Any],
        model_results: Dict[str, Any],
        include_charts: bool = True
    ) -> BytesIO:
        """
        Generate a complete PDF report.
        
        Parameters:
        -----------
        input_data : dict
            Input participant data
        predictions : dict
            Prediction results
        model_results : dict
            Model performance results
        include_charts : bool
            Whether to include charts
        
        Returns:
        --------
        BytesIO
            PDF buffer
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        story = []
        
        # Title
        story.append(Paragraph(self.title, self.styles['CustomTitle']))
        story.append(Spacer(1, 12))
        
        # Date
        date_str = datetime.now().strftime("%B %d, %Y")
        story.append(Paragraph(f"Generated: {date_str}", self.styles['CustomSubheading']))
        story.append(Spacer(1, 24))
        
        # ========== 1. Input Data ==========
        story.append(Paragraph("1. Participant Data", self.styles['CustomHeading']))
        story.append(Spacer(1, 8))
        
        input_df = pd.DataFrame([input_data])
        story.extend(self._create_table(input_df.T.reset_index().rename(
            columns={'index': 'Feature', 0: 'Value'}
        ), "Input Features"))
        
        story.append(Spacer(1, 12))
        
        # ========== 2. Prediction Results ==========
        story.append(Paragraph("2. Prediction Results", self.styles['CustomHeading']))
        story.append(Spacer(1, 8))
        
        if 'classification' in predictions:
            story.append(Paragraph("Classification", self.styles['CustomSubheading']))
            
            pred_class = predictions['classification']['predicted_class']
            confidence = predictions['classification'].get('confidence', None)
            
            class_desc = {
                'A': {'name': 'Excellent', 'color': '#1e3a5f'},
                'B': {'name': 'Good', 'color': '#10b981'},
                'C': {'name': 'Average', 'color': '#f59e0b'},
                'D': {'name': 'Needs Improvement', 'color': '#ef4444'}
            }
            
            desc = class_desc.get(pred_class, class_desc['D'])
            
            story.append(Paragraph(
                f"<b>Predicted Class:</b> {pred_class} - {desc['name']}",
                self.styles['CustomBody']
            ))
            
            if confidence:
                story.append(Paragraph(
                    f"<b>Confidence:</b> {confidence:.1%}",
                    self.styles['CustomBody']
                ))
            
            story.append(Spacer(1, 8))
        
        if 'regression' in predictions:
            story.append(Paragraph("Regression", self.styles['CustomSubheading']))
            story.append(Paragraph(
                f"<b>Predicted Broad Jump:</b> {predictions['regression']['predicted_value']:.1f} cm",
                self.styles['CustomBody']
            ))
            story.append(Spacer(1, 8))
        
        # ========== 3. Model Comparison ==========
        if model_results:
            story.append(Paragraph("3. Model Comparison", self.styles['CustomHeading']))
            story.append(Spacer(1, 8))
            
            if 'classification' in model_results:
                story.append(Paragraph("Classification Models", self.styles['CustomSubheading']))
                clf_df = pd.DataFrame(model_results['classification'])
                story.extend(self._create_table(clf_df, "Classification Performance"))
            
            if 'regression' in model_results:
                story.append(Paragraph("Regression Models", self.styles['CustomSubheading']))
                reg_df = pd.DataFrame(model_results['regression'])
                story.extend(self._create_table(reg_df, "Regression Performance"))
        
        # ========== 4. Recommendations ==========
        story.append(PageBreak())
        story.append(Paragraph("4. Recommendations", self.styles['CustomHeading']))
        story.append(Spacer(1, 8))
        
        # Class-specific recommendations
        if 'classification' in predictions:
            pred_class = predictions['classification']['predicted_class']
            recommendations = {
                'A': [
                    "Maintain current training regimen",
                    "Focus on injury prevention",
                    "Consider advanced performance metrics"
                ],
                'B': [
                    "Increase training intensity gradually",
                    "Focus on flexibility exercises",
                    "Monitor progress monthly"
                ],
                'C': [
                    "Structured training program recommended",
                    "Focus on core strength and flexibility",
                    "Aim for 20% improvement in 3 months"
                ],
                'D': [
                    "Begin with basic fitness program",
                    "Focus on consistency over intensity",
                    "Consult with fitness professional",
                    "Target 30% improvement in 6 months"
                ]
            }
            
            recs = recommendations.get(pred_class, recommendations['C'])
            story.append(Paragraph(f"<b>For Class {pred_class}:</b>", self.styles['CustomSubheading']))
            for rec in recs:
                story.append(Paragraph(f"• {rec}", self.styles['CustomBody']))
            story.append(Spacer(1, 12))
        
        # General recommendations
        story.append(Paragraph("General Recommendations", self.styles['CustomSubheading']))
        general_recs = [
            "Regular physical activity (at least 150 minutes/week)",
            "Balanced nutrition with adequate protein intake",
            "Adequate sleep (7-9 hours per night)",
            "Regular health check-ups including blood pressure monitoring"
        ]
        for rec in general_recs:
            story.append(Paragraph(f"• {rec}", self.styles['CustomBody']))
        
        # ========== 5. Disclaimer ==========
        story.append(Spacer(1, 24))
        story.append(Paragraph(
            "<i>Disclaimer: This report is generated by an AI/ML system for informational purposes only. "
            "It should not replace professional medical advice or fitness assessment.</i>",
            self.styles['CustomBody']
        ))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer


def generate_summary(
    input_data: Dict[str, Any],
    predictions: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a text summary of predictions.
    
    Parameters:
    -----------
    input_data : dict
        Input participant data
    predictions : dict
        Prediction results
    
    Returns:
    --------
    Dict
        Summary text and key metrics
    """
    summary = {
        'title': "Body Performance Analysis Summary",
        'key_findings': [],
        'metrics': {},
        'recommendations': []
    }
    
    # Extract key metrics
    age = input_data.get('age', 'N/A')
    gender = input_data.get('gender', 'N/A')
    
    summary['metrics']['Age'] = age
    summary['metrics']['Gender'] = gender
    
    if 'classification' in predictions:
        pred_class = predictions['classification']['predicted_class']
        confidence = predictions['classification'].get('confidence', None)
        
        class_desc = {
            'A': "Excellent performance - top tier",
            'B': "Good performance - above average",
            'C': "Average performance",
            'D': "Needs improvement - below average"
        }
        
        summary['key_findings'].append(f"Predicted Performance Class: {pred_class} - {class_desc[pred_class]}")
        if confidence:
            summary['key_findings'].append(f"Prediction Confidence: {confidence:.1%}")
    
    if 'regression' in predictions:
        pred_value = predictions['regression']['predicted_value']
        summary['key_findings'].append(f"Predicted Broad Jump Distance: {pred_value:.1f} cm")
        
        # Compare to average
        avg_jump = 190.13
        if pred_value > avg_jump + 20:
            summary['key_findings'].append(f"Above average explosive power (+{pred_value - avg_jump:.0f} cm)")
        elif pred_value < avg_jump - 20:
            summary['key_findings'].append(f"Below average explosive power ({pred_value - avg_jump:.0f} cm)")
    
    return summary


if __name__ == "__main__":
    # Test the module
    print("Testing report_generator module...")
    
    # Test sample
    generator = ReportGenerator()
    print("✅ ReportGenerator initialized")
    
    # Test summary generation
    test_input = {'age': 25, 'gender': 'M'}
    test_predictions = {
        'classification': {'predicted_class': 'B', 'confidence': 0.72},
        'regression': {'predicted_value': 195.5}
    }
    
    summary = generate_summary(test_input, test_predictions)
    print(f"✅ Summary generated: {len(summary['key_findings'])} findings")
    
    print("\n✅ report_generator.py loaded successfully")
