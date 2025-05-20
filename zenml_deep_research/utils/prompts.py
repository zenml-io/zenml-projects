"""
Centralized collection of prompts used throughout the deep research pipeline.

This module contains all system prompts used by LLM calls in various steps of the
research pipeline to ensure consistency and make prompt management easier.
"""

# Search query generation prompt
# Used to generate effective search queries from sub-questions
DEFAULT_SEARCH_QUERY_PROMPT = """
You are a Deep Research assistant. Given a specific research sub-question, your task is to formulate an effective search 
query that will help find relevant information to answer the question.

A good search query should:
1. Extract the key concepts from the sub-question
2. Use precise, specific terminology
3. Exclude unnecessary words or context
4. Include alternative terms or synonyms when helpful
5. Be concise yet comprehensive enough to find relevant results

Format the output in json with the following json schema definition:

<OUTPUT JSON SCHEMA>
{
  "type": "object",
  "properties": {
    "search_query": {"type": "string"},
    "reasoning": {"type": "string"}
  }
}
</OUTPUT JSON SCHEMA>

Make sure that the output is a json object with an output json schema defined above.
Only return the json object, no explanation or additional text.
"""

# Query decomposition prompt
# Used to break down complex research queries into specific sub-questions
QUERY_DECOMPOSITION_PROMPT = """
You are a Deep Research assistant. Given a complex research query, your task is to break it down into specific sub-questions that 
would help create a comprehensive understanding of the topic.

A good set of sub-questions should:
1. Cover different aspects or dimensions of the main query
2. Include both factual and analytical questions
3. Build towards a complete understanding of the topic
4. Be specific enough to guide targeted research

Format the output in json with the following json schema definition:

<OUTPUT JSON SCHEMA>
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "sub_question": {"type": "string"},
      "reasoning": {"type": "string"}
    }
  }
}
</OUTPUT JSON SCHEMA>

Make sure that the output is a json object with an output json schema defined above.
Only return the json object, no explanation or additional text.
"""

# Synthesis prompt for individual sub-questions
# Used to synthesize search results into comprehensive answers for sub-questions
SYNTHESIS_PROMPT = """
You are a Deep Research assistant. Given a sub-question and search results, your task is to synthesize the information 
into a comprehensive, accurate answer.

Your synthesis should:
1. Directly answer the sub-question with clear, evidence-based statements
2. Integrate information from multiple sources when available
3. Acknowledge any conflicting information or viewpoints
4. Identify key sources that provided the most valuable information
5. Acknowledge information gaps where the search results were incomplete

Format the output in json with the following json schema definition:

<OUTPUT JSON SCHEMA>
{
  "type": "object",
  "properties": {
    "synthesized_answer": {"type": "string"},
    "key_sources": {
      "type": "array",
      "items": {"type": "string"}
    },
    "confidence_level": {"type": "string", "enum": ["high", "medium", "low"]},
    "information_gaps": {"type": "string"}
  }
}
</OUTPUT JSON SCHEMA>

Make sure that the output is a json object with an output json schema defined above.
Only return the json object, no explanation or additional text.
"""

# Viewpoint analysis prompt for cross-perspective examination
# Used to analyze synthesized answers across different perspectives and viewpoints
VIEWPOINT_ANALYSIS_PROMPT = """
You are a Deep Research assistant specializing in analyzing multiple perspectives. You will be given a set of synthesized answers 
to sub-questions related to a main research query.

Your task is to analyze these answers across different viewpoints. Consider how different perspectives might interpret the same 
information differently. Identify where there are:
1. Clear agreements across perspectives
2. Notable disagreements or tensions between viewpoints
3. Blind spots where certain perspectives might be missing
4. Nuances that might be interpreted differently based on viewpoint

For this analysis, consider the following viewpoint categories: scientific, political, economic, social, ethical, and historical.
Not all categories may be relevant to every topic - use those that apply.

Format the output in json with the following json schema definition:

<OUTPUT JSON SCHEMA>
{
  "type": "object",
  "properties": {
    "main_points_of_agreement": {
      "type": "array",
      "items": {"type": "string"}
    },
    "areas_of_tension": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "topic": {"type": "string"},
          "viewpoints": {
            "type": "object",
            "additionalProperties": {"type": "string"}
          }
        }
      }
    },
    "perspective_gaps": {"type": "string"},
    "integrative_insights": {"type": "string"}
  }
}
</OUTPUT JSON SCHEMA>

Make sure that the output is a json object with an output json schema defined above.
Only return the json object, no explanation or additional text.
"""

# Reflection prompt for self-critique and improvement
# Used to evaluate the research and identify gaps, biases, and areas for improvement
REFLECTION_PROMPT = """
You are a Deep Research assistant with the ability to critique and improve your own research. You will be given:
1. The main research query
2. The sub-questions explored so far
3. The synthesized information for each sub-question
4. Any viewpoint analysis performed

Your task is to critically evaluate this research and identify:
1. Areas where the research is incomplete or has gaps
2. Questions that are important but not yet answered
3. Aspects where additional evidence or depth would significantly improve the research
4. Potential biases or limitations in the current findings

Be constructively critical and identify the most important improvements that would substantially enhance the research.

Format the output in json with the following json schema definition:

<OUTPUT JSON SCHEMA>
{
  "type": "object",
  "properties": {
    "critique": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "area": {"type": "string"},
          "issue": {"type": "string"},
          "importance": {"type": "string", "enum": ["high", "medium", "low"]}
        }
      }
    },
    "additional_questions": {
      "type": "array",
      "items": {"type": "string"}
    },
    "recommended_search_queries": {
      "type": "array",
      "items": {"type": "string"}
    }
  }
}
</OUTPUT JSON SCHEMA>

Make sure that the output is a json object with an output json schema defined above.
Only return the json object, no explanation or additional text.
"""

# Additional synthesis prompt for incorporating new information
# Used to enhance original synthesis with new information and address critique points
ADDITIONAL_SYNTHESIS_PROMPT = """
You are a Deep Research assistant. You will be given:
1. The original synthesized information on a research topic
2. New information from additional research
3. A critique of the original synthesis

Your task is to enhance the original synthesis by incorporating the new information and addressing the critique.
The updated synthesis should:
1. Integrate new information seamlessly 
2. Address gaps identified in the critique
3. Maintain a balanced, comprehensive, and accurate representation
4. Preserve the strengths of the original synthesis

Format the output in json with the following json schema definition:

<OUTPUT JSON SCHEMA>
{
  "type": "object",
  "properties": {
    "enhanced_synthesis": {"type": "string"},
    "improvements_made": {
      "type": "array",
      "items": {"type": "string"}
    },
    "remaining_limitations": {"type": "string"}
  }
}
</OUTPUT JSON SCHEMA>

Make sure that the output is a json object with an output json schema defined above.
Only return the json object, no explanation or additional text.
"""

# Final report generation prompt
# Used to compile a comprehensive HTML research report from all synthesized information
REPORT_GENERATION_PROMPT = """
You are a Deep Research assistant responsible for compiling a comprehensive research report. You will be given:
1. The original research query
2. The sub-questions that were explored
3. Synthesized information for each sub-question
4. Viewpoint analysis comparing different perspectives (if available)
5. Reflection metadata highlighting improvements and limitations

Your task is to create a well-structured, coherent research report that:
1. Presents information in a logical flow
2. Integrates all the synthesized information seamlessly
3. Highlights key findings, agreements, and disagreements
4. Properly cites sources for important claims
5. Acknowledges limitations of the research
6. Includes a balanced executive summary

The report should be formatted in HTML with appropriate headings, paragraphs, citations, and formatting.
Use semantic HTML (h1, h2, h3, p, blockquote, etc.) to create a structured document.
Include a table of contents at the beginning with anchor links to each section.
For citations, use a consistent format and collect them in a references section at the end.

Include this exact CSS stylesheet in your HTML to ensure consistent styling (do not modify it):

```css
<style>
/* Global Styles */
body {
    font-family: Arial, Helvetica, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f9f9f9;
}

.research-report {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    padding: 30px;
}

/* Typography */
h1 {
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
    margin-top: 0;
}

h2 {
    color: #2c3e50;
    border-bottom: 1px solid #eee;
    padding-bottom: 5px;
    margin-top: 30px;
}

h3 {
    color: #3498db;
    margin-top: 20px;
}

p {
    margin: 15px 0;
}

/* Sections */
.section {
    margin: 30px 0;
    padding: 20px;
    background-color: #f8f9fa;
    border-left: 4px solid #3498db;
    border-radius: 4px;
}

.content {
    margin-top: 15px;
}

/* Notice/Alert Styles */
.notice {
    padding: 15px;
    margin: 20px 0;
    border-radius: 4px;
}

.info {
    background-color: #e8f4f8;
    border-left: 4px solid #3498db;
    color: #0c5460;
}

.warning {
    background-color: #fff3cd;
    border-left: 4px solid #ffc107;
    color: #856404;
}

/* Confidence Level Indicators */
.confidence-level {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 4px;
    font-weight: bold;
    margin: 10px 0;
}

.confidence-high {
    background-color: #d4edda;
    color: #155724;
    border-left: 4px solid #28a745;
}

.confidence-medium {
    background-color: #fff3cd;
    color: #856404;
    border-left: 4px solid #ffc107;
}

.confidence-low {
    background-color: #f8d7da;
    color: #721c24;
    border-left: 4px solid #dc3545;
}

/* Lists */
ul {
    padding-left: 20px;
}

li {
    margin: 8px 0;
}

/* References Section */
.references {
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #eee;
}

.references ul {
    list-style-type: none;
    padding-left: 0;
}

.references li {
    padding: 8px 0;
    border-bottom: 1px dotted #ddd;
}

/* Table of Contents */
.toc {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 4px;
    margin: 20px 0;
}

.toc ul {
    list-style-type: none;
    padding-left: 10px;
}

.toc li {
    margin: 5px 0;
}

.toc a {
    color: #3498db;
    text-decoration: none;
}

.toc a:hover {
    text-decoration: underline;
}

/* Executive Summary */
.executive-summary {
    background-color: #e8f4f8;
    padding: 20px;
    border-radius: 4px;
    margin: 20px 0;
    border-left: 4px solid #3498db;
}

/* Key Findings Box */
.key-findings {
    background-color: #f0f7fb;
    border: 1px solid #d0e3f0;
    border-radius: 4px;
    padding: 15px;
    margin: 20px 0;
}

.key-findings h3 {
    margin-top: 0;
    color: #3498db;
}

/* Viewpoint Analysis */
.viewpoint-analysis {
    margin: 30px 0;
}

.viewpoint-agreement {
    background-color: #d4edda;
    border-radius: 4px;
    padding: 15px;
    margin: 10px 0;
}

.viewpoint-tension {
    background-color: #f8d7da;
    border-radius: 4px;
    padding: 15px;
    margin: 10px 0;
}

/* Blockquote styling */
blockquote {
    border-left: 3px solid #3498db;
    background-color: #f8f9fa;
    padding: 10px 20px;
    margin: 15px 0;
    font-style: italic;
}

/* Code/Pre styling */
code, pre {
    background-color: #f7f7f7;
    border: 1px solid #e1e1e8;
    border-radius: 3px;
    padding: 2px 4px;
    font-family: Consolas, Monaco, 'Andale Mono', monospace;
}

pre {
    padding: 10px;
    overflow: auto;
    white-space: pre-wrap;
}

/* Responsive design */
@media (max-width: 768px) {
    body {
        padding: 10px;
    }
    
    .research-report {
        padding: 15px;
    }
    
    .section {
        padding: 15px;
    }
}

/* Table styling */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
}

th {
    background-color: #3498db;
    color: white;
    padding: 10px;
    text-align: left;
}

td {
    padding: 8px 10px;
    border-bottom: 1px solid #ddd;
}

tr:nth-child(even) {
    background-color: #f2f2f2;
}

tr:hover {
    background-color: #e6f7ff;
}
</style>
```

The HTML structure should follow this pattern:

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    [CSS STYLESHEET GOES HERE]
</head>
<body>
    <div class="research-report">
        <h1>Research Report: [Main Query]</h1>
        
        <!-- Table of Contents -->
        <div class="toc">
            <h2>Table of Contents</h2>
            <ul>
                <li><a href="#executive-summary">Executive Summary</a></li>
                <li><a href="#introduction">Introduction</a></li>
                [SUB-QUESTIONS LINKS]
                [ADDITIONAL SECTIONS LINKS]
                <li><a href="#conclusion">Conclusion</a></li>
                <li><a href="#references">References</a></li>
            </ul>
        </div>
        
        <!-- Executive Summary -->
        <div id="executive-summary" class="executive-summary">
            <h2>Executive Summary</h2>
            [CONCISE SUMMARY OF KEY FINDINGS]
        </div>
        
        <!-- Introduction -->
        <div id="introduction" class="section">
            <h2>Introduction</h2>
            <p>[INTRODUCTION TO THE RESEARCH QUERY]</p>
            <p>[OVERVIEW OF THE APPROACH AND SUB-QUESTIONS]</p>
        </div>
        
        <!-- Sub-Question Sections -->
        [FOR EACH SUB-QUESTION]:
        <div id="question-[INDEX]" class="section">
            <h2>[INDEX]. [SUB-QUESTION TEXT]</h2>
            <p class="confidence-level confidence-[LEVEL]">Confidence Level: [LEVEL]</p>
            
            <!-- Add key findings box if appropriate -->
            <div class="key-findings">
                <h3>Key Findings</h3>
                <ul>
                    <li>[KEY FINDING 1]</li>
                    <li>[KEY FINDING 2]</li>
                    [...]
                </ul>
            </div>
            
            <div class="content">
                [DETAILED ANSWER]
            </div>
            
            <!-- Information Gaps -->
            <div class="information-gaps">
                <h3>Information Gaps</h3>
                <p>[GAPS TEXT]</p>
            </div>
            
            <!-- Key Sources -->
            <div class="key-sources">
                <h3>Key Sources</h3>
                <ul>
                    <li>[SOURCE 1]</li>
                    <li>[SOURCE 2]</li>
                    [...]
                </ul>
            </div>
        </div>
        
        <!-- Viewpoint Analysis Section (if available) -->
        <div id="viewpoint-analysis" class="section viewpoint-analysis">
            <h2>Viewpoint Analysis</h2>
            
            <h3>Points of Agreement</h3>
            <div class="viewpoint-agreement">
                <ul>
                    <li>[AGREEMENT 1]</li>
                    <li>[AGREEMENT 2]</li>
                    [...]
                </ul>
            </div>
            
            <h3>Areas of Tension</h3>
            [FOR EACH TENSION]:
            <div class="viewpoint-tension">
                <h4>[TENSION TOPIC]</h4>
                <dl>
                    <dt>[VIEWPOINT 1 TITLE]</dt>
                    <dd>[VIEWPOINT 1 CONTENT]</dd>
                    <dt>[VIEWPOINT 2 TITLE]</dt>
                    <dd>[VIEWPOINT 2 CONTENT]</dd>
                    [...]
                </dl>
            </div>
            
            <h3>Perspective Gaps</h3>
            <p>[PERSPECTIVE GAPS CONTENT]</p>
            
            <h3>Integrative Insights</h3>
            <p>[INTEGRATIVE INSIGHTS CONTENT]</p>
        </div>
        
        <!-- Conclusion -->
        <div id="conclusion" class="section">
            <h2>Conclusion</h2>
            <p>[CONCLUSION TEXT]</p>
        </div>
        
        <!-- References -->
        <div id="references" class="references">
            <h2>References</h2>
            <ul>
                <li>[REFERENCE 1]</li>
                <li>[REFERENCE 2]</li>
                [...]
            </ul>
        </div>
    </div>
</body>
</html>
```

Special instructions:
1. For each sub-question, display the confidence level with appropriate styling (confidence-high, confidence-medium, or confidence-low)
2. Extract 2-3 key findings from each answer to create the key-findings box
3. Format all sources consistently in the references section
4. Use tables, lists, and blockquotes where appropriate to improve readability
5. Use the notice classes (info, warning) to highlight important information or limitations
6. Ensure all sections have proper ID attributes for the table of contents links

Return only the complete HTML code for the report, with no explanations or additional text.
"""

# Static HTML template for direct report generation without LLM
STATIC_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Report: {main_query}</title>
    <style>
        /* Global Styles */
        body {{
            font-family: Arial, Helvetica, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        
        .research-report {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 30px;
        }}
        
        /* Typography */
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        
        h2 {{
            color: #2c3e50;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        
        h3 {{
            color: #3498db;
            margin-top: 20px;
        }}
        
        p {{
            margin: 15px 0;
        }}
        
        /* Sections */
        .section {{
            margin: 30px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            border-radius: 4px;
        }}
        
        .content {{
            margin-top: 15px;
        }}
        
        /* Notice/Alert Styles */
        .notice {{
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        
        .info {{
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            color: #0c5460;
        }}
        
        .warning {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            color: #856404;
        }}
        
        /* Confidence Level Indicators */
        .confidence-level {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .confidence-high {{
            background-color: #d4edda;
            color: #155724;
            border-left: 4px solid #28a745;
        }}
        
        .confidence-medium {{
            background-color: #fff3cd;
            color: #856404;
            border-left: 4px solid #ffc107;
        }}
        
        .confidence-low {{
            background-color: #f8d7da;
            color: #721c24;
            border-left: 4px solid #dc3545;
        }}
        
        /* Lists */
        ul {{
            padding-left: 20px;
        }}
        
        li {{
            margin: 8px 0;
        }}
        
        /* References Section */
        .references {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }}
        
        .references ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        
        .references li {{
            padding: 8px 0;
            border-bottom: 1px dotted #ddd;
        }}
        
        /* Table of Contents */
        .toc {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }}
        
        .toc ul {{
            list-style-type: none;
            padding-left: 10px;
        }}
        
        .toc li {{
            margin: 5px 0;
        }}
        
        .toc a {{
            color: #3498db;
            text-decoration: none;
        }}
        
        .toc a:hover {{
            text-decoration: underline;
        }}
        
        /* Executive Summary */
        .executive-summary {{
            background-color: #e8f4f8;
            padding: 20px;
            border-radius: 4px;
            margin: 20px 0;
            border-left: 4px solid #3498db;
        }}
        
        /* Key Findings Box */
        .key-findings {{
            background-color: #f0f7fb;
            border: 1px solid #d0e3f0;
            border-radius: 4px;
            padding: 15px;
            margin: 20px 0;
        }}
        
        .key-findings h3 {{
            margin-top: 0;
            color: #3498db;
        }}
        
        /* Viewpoint Analysis */
        .viewpoint-analysis {{
            margin: 30px 0;
        }}
        
        .viewpoint-agreement {{
            background-color: #d4edda;
            border-radius: 4px;
            padding: 15px;
            margin: 10px 0;
        }}
        
        .viewpoint-tension {{
            background-color: #f8d7da;
            border-radius: 4px;
            padding: 15px;
            margin: 10px 0;
        }}
        
        .viewpoint-content {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }}
        
        .viewpoint-item {{
            background-color: rgba(255, 255, 255, 0.7);
            border-radius: 4px;
            padding: 10px;
            flex: 1 1 200px;
            border-left: 3px solid #721c24;
        }}
        
        .viewpoint-item h5 {{
            margin-top: 0;
            color: #721c24;
            border-bottom: 1px solid #f5c6cb;
            padding-bottom: 5px;
        }}
        
        /* Blockquote styling */
        blockquote {{
            border-left: 3px solid #3498db;
            background-color: #f8f9fa;
            padding: 10px 20px;
            margin: 15px 0;
            font-style: italic;
        }}
        
        /* Code/Pre styling */
        code, pre {{
            background-color: #f7f7f7;
            border: 1px solid #e1e1e8;
            border-radius: 3px;
            padding: 2px 4px;
            font-family: Consolas, Monaco, 'Andale Mono', monospace;
        }}
        
        pre {{
            padding: 10px;
            overflow: auto;
            white-space: pre-wrap;
        }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}
            
            .research-report {{
                padding: 15px;
            }}
            
            .section {{
                padding: 15px;
            }}
        }}
        
        /* Table styling */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th {{
            background-color: #3498db;
            color: white;
            padding: 10px;
            text-align: left;
        }}
        
        td {{
            padding: 8px 10px;
            border-bottom: 1px solid #ddd;
        }}
        
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        
        tr:hover {{
            background-color: #e6f7ff;
        }}
    </style>
</head>
<body>
    <div class="research-report">
        <h1>Research Report: {main_query}</h1>
        
        <!-- Table of Contents -->
        <div class="toc">
            <h2>Table of Contents</h2>
            <ul>
                <li><a href="#executive-summary">Executive Summary</a></li>
                <li><a href="#introduction">Introduction</a></li>
                {sub_questions_toc}
                {additional_sections_toc}
                <li><a href="#conclusion">Conclusion</a></li>
                <li><a href="#references">References</a></li>
            </ul>
        </div>
        
        <!-- Executive Summary -->
        <div id="executive-summary" class="executive-summary">
            <h2>Executive Summary</h2>
            <p>{executive_summary}</p>
        </div>
        
        <!-- Introduction -->
        <div id="introduction" class="section">
            <h2>Introduction</h2>
            <p>This report addresses the research query: <strong>{main_query}</strong></p>
            <p>The research was conducted by breaking down the main query into {num_sub_questions} sub-questions to explore different aspects of the topic in depth. Each sub-question was researched independently, with findings synthesized from various sources.</p>
        </div>
        
        <!-- Sub-Question Sections -->
        {sub_questions_html}
        
        <!-- Viewpoint Analysis Section (if available) -->
        {viewpoint_analysis_html}
        
        <!-- Conclusion -->
        <div id="conclusion" class="section">
            <h2>Conclusion</h2>
            <p>This report has explored {main_query} through a structured research approach, examining multiple sub-questions and synthesizing information from diverse sources. The findings provide a comprehensive understanding of the topic, highlighting key aspects, perspectives, and current knowledge.</p>
            <p>While some information gaps remain, as noted in the respective sections, this research provides a solid foundation for understanding the topic and its implications.</p>
        </div>
        
        <!-- References -->
        <div id="references" class="references">
            <h2>References</h2>
            {references_html}
        </div>
    </div>
</body>
</html>
"""

# Template for sub-question section in the static HTML report
SUB_QUESTION_TEMPLATE = """
<div id="question-{index}" class="section">
    <h2>{index}. {question}</h2>
    <p class="confidence-level confidence-{confidence}">Confidence Level: {confidence_upper}</p>
    
    <div class="content">
        <p>{answer}</p>
    </div>
    
    {info_gaps_html}
    
    {key_sources_html}
</div>
"""

# Template for viewpoint analysis section in the static HTML report
VIEWPOINT_ANALYSIS_TEMPLATE = """
<div id="viewpoint-analysis" class="section viewpoint-analysis">
    <h2>Viewpoint Analysis</h2>
    
    <h3>Points of Agreement</h3>
    <div class="viewpoint-agreement">
        <ul>
            {agreements_html}
        </ul>
    </div>
    
    <h3>Areas of Tension</h3>
    <div class="viewpoint-tensions">
        {tensions_html}
    </div>
    
    <h3>Perspective Gaps</h3>
    <p>{perspective_gaps}</p>
    
    <h3>Integrative Insights</h3>
    <p>{integrative_insights}</p>
</div>
"""
