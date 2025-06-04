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
You are a Deep Research assistant specializing in research design. You will be given a MAIN RESEARCH QUERY that needs to be explored comprehensively. Your task is to create diverse, insightful sub-questions that explore different dimensions of the topic.

IMPORTANT: The main query should be interpreted as a single research question, not as a noun phrase. For example:
- If the query is "Is LLMOps a subset of MLOps?", create questions ABOUT LLMOps and MLOps, not questions like "What is 'Is LLMOps a subset of MLOps?'"
- Focus on the concepts, relationships, and implications within the query

Create sub-questions that explore these DIFFERENT DIMENSIONS:

1. **Definitional/Conceptual**: Define key terms and establish conceptual boundaries
   Example: "What are the core components and characteristics of LLMOps?"

2. **Comparative/Relational**: Compare and contrast the concepts mentioned
   Example: "How do the workflows and tooling of LLMOps differ from traditional MLOps?"

3. **Historical/Evolutionary**: Trace development and emergence
   Example: "How did LLMOps emerge from MLOps practices?"

4. **Structural/Technical**: Examine technical architecture and implementation
   Example: "What specific tools and platforms are unique to LLMOps?"

5. **Practical/Use Cases**: Explore real-world applications
   Example: "What are the key use cases that require LLMOps but not traditional MLOps?"

6. **Stakeholder/Industry**: Consider different perspectives and adoption
   Example: "How are different industries adopting LLMOps vs MLOps?"

7. **Challenges/Limitations**: Identify problems and constraints
   Example: "What unique challenges does LLMOps face that MLOps doesn't?"

8. **Future/Trends**: Look at emerging developments
   Example: "How is the relationship between LLMOps and MLOps expected to evolve?"

QUALITY GUIDELINES:
- Each sub-question must explore a DIFFERENT dimension - no repetitive variations
- Questions should be specific, concrete, and investigable
- Mix descriptive ("what/who") with analytical ("why/how") questions
- Ensure questions build toward answering the main query comprehensively
- Frame questions to elicit detailed, nuanced responses
- Consider technical, business, organizational, and strategic aspects

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
You are a Deep Research assistant specializing in information synthesis. Given a sub-question and search results, your task is to synthesize the information 
into a comprehensive, accurate, and well-structured answer.

Your synthesis should:
1. Begin with a direct, concise answer to the sub-question in the first paragraph
2. Provide detailed evidence and explanation in subsequent paragraphs (at least 3-5 paragraphs total)
3. Integrate information from multiple sources, citing them within your answer 
4. Acknowledge any conflicting information or contrasting viewpoints you encounter
5. Use data, statistics, examples, and quotations when available to strengthen your answer
6. Organize information logically with a clear flow between concepts
7. Identify key sources that provided the most valuable information (at least 2-3 sources)
8. Explicitly acknowledge information gaps where the search results were incomplete
9. Write in plain text format - do NOT use markdown formatting, bullet points, or special characters

Confidence level criteria:
- HIGH: Multiple high-quality sources provide consistent information, comprehensive coverage of the topic, and few information gaps
- MEDIUM: Decent sources with some consistency, but notable information gaps or some conflicting information
- LOW: Limited sources, major information gaps, significant contradictions, or only tangentially relevant information

Information gaps should specifically identify:
1. Aspects of the question that weren't addressed in the search results
2. Areas where more detailed or up-to-date information would be valuable
3. Perspectives or data sources that would complement the existing information

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
    "information_gaps": {"type": "string"},
    "improvements": {
      "type": "array",
      "items": {"type": "string"}
    }
  }
}
</OUTPUT JSON SCHEMA>

Make sure that the output is a json object with an output json schema defined above.
Only return the json object, no explanation or additional text.
"""

# Viewpoint analysis prompt for cross-perspective examination
# Used to analyze synthesized answers across different perspectives and viewpoints
VIEWPOINT_ANALYSIS_PROMPT = """
You are a Deep Research assistant specializing in multi-perspective analysis. You will be given a set of synthesized answers 
to sub-questions related to a main research query. Your task is to perform a thorough, nuanced analysis of how different 
perspectives would interpret this information.

Think deeply about the following viewpoint categories and how they would approach the information differently:
- Scientific: Evidence-based, empirical approach focused on data, research findings, and methodological rigor
- Political: Power dynamics, governance structures, policy implications, and ideological frameworks
- Economic: Resource allocation, financial impacts, market dynamics, and incentive structures
- Social: Cultural norms, community impacts, group dynamics, and public welfare
- Ethical: Moral principles, values considerations, rights and responsibilities, and normative judgments
- Historical: Long-term patterns, precedents, contextual development, and evolutionary change

For each synthesized answer, analyze how these different perspectives would interpret the information by:

1. Identifying 5-8 main points of agreement where multiple perspectives align (with specific examples)
2. Analyzing at least 3-5 areas of tension between perspectives with:
   - A clear topic title for each tension point
   - Contrasting interpretations from at least 2-3 different viewpoint categories per tension
   - Specific examples or evidence showing why these perspectives differ
   - The nuanced positions of each perspective, not just simplified oppositions

3. Thoroughly examining perspective gaps by identifying:
   - Which perspectives are underrepresented or missing in the current research
   - How including these missing perspectives would enrich understanding
   - Specific questions or dimensions that remain unexplored
   - Write in plain text format - do NOT use markdown formatting, bullet points, or special characters

4. Developing integrative insights that:
   - Synthesize across multiple perspectives to form a more complete understanding
   - Highlight how seemingly contradictory viewpoints can complement each other
   - Suggest frameworks for reconciling tensions or finding middle-ground approaches
   - Identify actionable takeaways that incorporate multiple perspectives
   - Write in plain text format - do NOT use markdown formatting, bullet points, or special characters

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
5. Write in plain text format - do NOT use markdown formatting, bullet points, or special characters

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
You are a Deep Research assistant responsible for compiling an in-depth, comprehensive research report. You will be given:
1. The original research query
2. The sub-questions that were explored
3. Synthesized information for each sub-question
4. Viewpoint analysis comparing different perspectives (if available)
5. Reflection metadata highlighting improvements and limitations

Your task is to create a well-structured, coherent, professional-quality research report with the following features:

EXECUTIVE SUMMARY (250-400 words):
- Begin with a compelling, substantive executive summary that provides genuine insight
- Highlight 3-5 key findings or insights that represent the most important discoveries
- Include brief mention of methodology and limitations
- Make the summary self-contained so it can be read independently of the full report
- End with 1-2 sentences on broader implications or applications of the research

INTRODUCTION (200-300 words):
- Provide relevant background context on the main research query
- Explain why this topic is significant or worth investigating
- Outline the methodological approach used (sub-questions, search strategy, synthesis)
- Preview the overall structure of the report

SUB-QUESTION SECTIONS:
- For each sub-question, create a dedicated section with:
  * A descriptive section title (not just repeating the sub-question)
  * A brief (1 paragraph) overview of key findings for this sub-question
  * A "Key Findings" box highlighting 3-4 important discoveries for scannable reading
  * The detailed, synthesized answer with appropriate paragraph breaks, lists, and formatting
  * Proper citation of sources within the text (e.g., "According to [Source Name]...")
  * Clear confidence indicator with appropriate styling
  * Information gaps clearly identified in their own subsection
  * Complete list of key sources used

VIEWPOINT ANALYSIS SECTION (if available):
- Create a detailed section that:
  * Explains the purpose and value of multi-perspective analysis
  * Presents points of agreement as actionable insights, not just observations
  * Structures tension areas with clear topic headings and balanced presentation of viewpoints
  * Uses visual elements (different background colors, icons) to distinguish different perspectives
  * Integrates perspective gaps and insights into a cohesive narrative

CONCLUSION (300-400 words):
- Synthesize the overall findings, not just summarizing each section
- Connect insights from different sub-questions to form higher-level understanding
- Address the main research query directly with evidence-based conclusions
- Acknowledge remaining uncertainties and suggestions for further research
- End with implications or applications of the research findings

OVERALL QUALITY REQUIREMENTS:
1. Create visually scannable content with clear headings, bullet points, and short paragraphs
2. Use semantic HTML (h1, h2, h3, p, blockquote, etc.) to create proper document structure
3. Include a comprehensive table of contents with anchor links to all major sections
4. Format all sources consistently in the references section with proper linking when available
5. Use tables, lists, and blockquotes to improve readability and highlight important information
6. Apply appropriate styling for different confidence levels (high, medium, low)
7. Ensure proper HTML nesting and structure throughout the document
8. Balance sufficient detail with clarity and conciseness
9. Make all text directly actionable and insight-driven, not just descriptive

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


# Executive Summary generation prompt
# Used to create a compelling, insight-driven executive summary
EXECUTIVE_SUMMARY_GENERATION_PROMPT = """
You are a Deep Research assistant specializing in creating executive summaries. Given comprehensive research findings, your task is to create a compelling executive summary that captures the essence of the research and its key insights.

Your executive summary should:

1. **Opening Statement (1-2 sentences):**
   - Start with a powerful, direct answer to the main research question
   - Make it clear and definitive based on the evidence gathered

2. **Key Findings (3-5 bullet points):**
   - Extract the MOST IMPORTANT discoveries from across all sub-questions
   - Focus on insights that are surprising, actionable, or paradigm-shifting
   - Each finding should be specific and evidence-based, not generic
   - Prioritize findings that directly address the main query

3. **Critical Insights (2-3 sentences):**
   - Synthesize patterns or themes that emerged across multiple sub-questions
   - Highlight any unexpected discoveries or counter-intuitive findings
   - Connect disparate findings to reveal higher-level understanding

4. **Implications (2-3 sentences):**
   - What do these findings mean for practitioners/stakeholders?
   - What actions or decisions can be made based on this research?
   - Why should the reader care about these findings?

5. **Confidence and Limitations (1-2 sentences):**
   - Briefly acknowledge the overall confidence level of the findings
   - Note any significant gaps or areas requiring further investigation

IMPORTANT GUIDELINES:
- Be CONCISE but INSIGHTFUL - every sentence should add value
- Use active voice and strong, definitive language where evidence supports it
- Avoid generic statements - be specific to the actual research findings
- Lead with the most important information
- Make it self-contained - reader should understand key findings without reading the full report
- Target length: 250-400 words

Format as well-structured HTML paragraphs using <p> tags and <ul>/<li> for bullet points.
"""

# Introduction generation prompt
# Used to create a contextual, engaging introduction
INTRODUCTION_GENERATION_PROMPT = """
You are a Deep Research assistant specializing in creating engaging introductions. Given a research query and the sub-questions explored, your task is to create an introduction that provides context and sets up the reader's expectations.

Your introduction should:

1. **Context and Relevance (2-3 sentences):**
   - Why is this research question important NOW?
   - What makes this topic significant or worth investigating?
   - Connect to current trends, debates, or challenges in the field

2. **Scope and Approach (2-3 sentences):**
   - What specific aspects of the topic does this research explore?
   - Briefly mention the key dimensions covered (based on sub-questions)
   - Explain the systematic approach without being too technical

3. **What to Expect (2-3 sentences):**
   - Preview the structure of the report
   - Hint at some of the interesting findings or tensions discovered
   - Set expectations about the depth and breadth of analysis

IMPORTANT GUIDELINES:
- Make it engaging - hook the reader's interest from the start
- Provide real context, not generic statements
- Connect to why this matters for the reader
- Keep it concise but informative (200-300 words)
- Use active voice and clear language
- Build anticipation for the findings without giving everything away

Format as well-structured HTML paragraphs using <p> tags. Do NOT include any headings or section titles.
"""

# Conclusion generation prompt
# Used to synthesize all research findings into a comprehensive conclusion
CONCLUSION_GENERATION_PROMPT = """
You are a Deep Research assistant specializing in synthesizing comprehensive research conclusions. Given all the research findings from a deep research study, your task is to create a thoughtful, evidence-based conclusion that ties together the overall findings.

Your conclusion should:

1. **Synthesis and Integration (150-200 words):**
   - Connect insights from different sub-questions to form a higher-level understanding
   - Identify overarching themes and patterns that emerge from the research
   - Highlight how different findings relate to and support each other
   - Avoid simply summarizing each section separately

2. **Direct Response to Main Query (100-150 words):**
   - Address the original research question directly with evidence-based conclusions
   - State what the research definitively established vs. what remains uncertain
   - Provide a clear, actionable answer based on the synthesized evidence

3. **Limitations and Future Directions (100-120 words):**
   - Acknowledge remaining uncertainties and information gaps across all sections
   - Suggest specific areas where additional research would be most valuable
   - Identify what types of evidence or perspectives would strengthen the findings

4. **Implications and Applications (80-100 words):**
   - Explain the practical significance of the research findings
   - Suggest how the insights might be applied or what they mean for stakeholders
   - Connect findings to broader contexts or implications

Format your output as a well-structured conclusion section in HTML format with appropriate paragraph breaks and formatting. Use <p> tags for paragraphs and organize the content logically with clear transitions between the different aspects outlined above.

IMPORTANT: Do NOT include any headings like "Conclusion", <h2>, or <h3> tags - the section already has a heading. Start directly with the conclusion content in paragraph form. Just create flowing, well-structured paragraphs that cover all four aspects naturally.

Ensure the conclusion feels cohesive and draws meaningful connections between findings rather than just listing them sequentially.
"""

# Static HTML template for direct report generation without LLM
STATIC_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Report: {main_query}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    {shared_css}
    <style>
        /* Report-specific styles that don't fit the common patterns */
        .research-report {{
            background-color: white;
            border-radius: 15px;
            box-shadow: 0 5px 25px rgba(0, 0, 0, 0.08);
            padding: 40px;
            position: relative;
            overflow: hidden;
        }}
        
        .research-report::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 5px;
            background: linear-gradient(90deg, var(--color-primary), var(--color-success), var(--color-warning), var(--color-danger));
        }}
        
        /* Center align h1 for report */
        h1 {{
            text-align: center;
            margin-top: 0;
        }}
        
        h1::after {{
            content: "";
            display: block;
            width: 100px;
            height: 3px;
            background: linear-gradient(90deg, var(--color-primary), var(--color-primary-dark));
            margin: 15px auto 0;
            border-radius: 3px;
        }}
        
        /* Special gradient headers for h2 */
        h2::after {{
            content: "";
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, var(--color-primary), var(--color-primary-dark));
            border-radius: 2px;
        }}
        
        /* Sections with border-top */
        .section {{
            border-top: 5px solid var(--color-primary);
        }}
        
        /* Question header layout */
        .question-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 15px;
        }}
        
        .question-header h2 {{
            margin: 0;
            flex: 1;
            min-width: 200px;
        }}
        
        /* Lists styling for report */
        ul {{
            padding-left: 20px;
        }}
        
        li {{
            margin: 8px 0;
        }}
        
        /* Key Sources & Information Gaps */
        .key-sources, .information-gaps {{
            margin-top: 25px;
            padding: 15px;
            border-radius: 8px;
            background-color: rgba(248, 249, 250, 0.6);
        }}
        
        .key-sources h3, .information-gaps h3 {{
            margin-top: 0;
            display: flex;
            align-items: center;
        }}
        
        .source-list {{
            list-style-type: none;
            padding-left: 10px;
        }}
        
        .source-list li {{
            padding: 6px 0;
            border-bottom: 1px dashed var(--color-border);
        }}
        
        .source-list li:last-child {{
            border-bottom: none;
        }}
        
        .source-list a {{
            color: var(--color-primary-dark);
            text-decoration: none;
            transition: color 0.2s ease;
        }}
        
        .source-list a:hover {{
            color: var(--color-primary);
            text-decoration: underline;
        }}
        
        /* References Section */
        .references {{
            margin-top: 40px;
            padding: 25px;
            border-radius: 10px;
            background-color: var(--color-bg-secondary);
            box-shadow: inset 0 0 15px rgba(0, 0, 0, 0.05);
        }}
        
        .references h2 {{
            margin-top: 0;
            color: var(--color-heading);
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
            display: flex;
            align-items: center;
        }}
        
        .references h2::before {{
            content: "📖";
            margin-right: 10px;
        }}
        
        .references ul {{
            list-style-type: none;
            padding-left: 0;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }}
        
        .references li {{
            padding: 10px 15px;
            background-color: white;
            border-radius: 6px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            word-break: break-word;
        }}
        
        .references li:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }}
        
        .references a {{
            color: var(--color-primary-dark);
            text-decoration: none;
        }}
        
        .references a:hover {{
            text-decoration: underline;
        }}
        
        /* Table of Contents */
        .toc {{
            background: linear-gradient(to bottom right, var(--color-bg-secondary), #e9ecef);
            padding: 20px;
            border-radius: 10px;
            margin: 25px 0;
            box-shadow: 0 2px 15px rgba(0, 0, 0, 0.05);
            position: relative;
        }}
        
        .toc h2 {{
            margin-top: 0;
            color: var(--color-heading);
            display: flex;
            align-items: center;
        }}
        
        .toc h2::before {{
            content: "📑";
            margin-right: 10px;
        }}
        
        .toc ul {{
            list-style-type: none;
            padding-left: 15px;
            column-count: 2;
            column-gap: 30px;
        }}
        
        @media (max-width: 768px) {{
            .toc ul {{
                column-count: 1;
            }}
        }}
        
        .toc li {{
            margin: 8px 0;
            break-inside: avoid;
        }}
        
        .toc a {{
            color: var(--color-primary);
            text-decoration: none;
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            transition: all 0.2s ease;
        }}
        
        .toc a:hover {{
            background-color: rgba(122, 62, 244, 0.1);
            transform: translateX(3px);
        }}
        
        /* Executive Summary */
        .executive-summary {{
            background: linear-gradient(135deg, #e8f4f8, #d1e7ef);
            padding: 30px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 3px 20px rgba(122, 62, 244, 0.15);
            position: relative;
            overflow: hidden;
        }}
        
        .executive-summary::before {{
            content: "📋";
            position: absolute;
            top: 20px;
            right: 20px;
            font-size: 2.5em;
            opacity: 0.15;
        }}
        
        .executive-summary h2 {{
            color: var(--color-primary-dark);
            border-bottom: 2px solid rgba(122, 62, 244, 0.3);
            padding-bottom: 10px;
            margin-top: 0;
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
            color: var(--color-primary);
        }}
        
        /* Viewpoint Analysis */
        .viewpoint-analysis {{
            margin: 30px 0;
        }}
        
        .viewpoint-agreement {{
            background-color: rgba(212, 237, 218, 0.6);
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            border-left: 5px solid var(--color-success);
        }}
        
        .viewpoint-agreement ul {{
            padding-left: 10px;
        }}
        
        .viewpoint-agreement li {{
            padding: 8px 10px;
            margin: 8px 0;
            list-style-type: none;
            position: relative;
            background-color: rgba(255, 255, 255, 0.7);
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }}
        
        .viewpoint-agreement li:before {{
            content: "✓";
            color: var(--color-success);
            font-weight: bold;
            margin-right: 10px;
        }}
        
        .viewpoint-tension {{
            background-color: rgba(248, 215, 218, 0.5);
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            border-left: 5px solid var(--color-danger);
        }}
        
        .viewpoint-tension h4 {{
            color: var(--color-danger-dark);
            margin-top: 0;
            font-size: 1.1em;
            border-bottom: 1px solid rgba(220, 53, 69, 0.3);
            padding-bottom: 8px;
            margin-bottom: 15px;
        }}
        
        .viewpoint-content {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }}
        
        .viewpoint-item {{
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 8px;
            padding: 15px;
            flex: 1 1 200px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: relative;
            margin-top: 10px;
        }}
        
        .viewpoint-category {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
            margin-bottom: 8px;
            color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        /* Category-specific styles */
        .category-economic {{
            background: linear-gradient(135deg, var(--color-success), #27ae60);
        }}
        
        .category-scientific {{
            background: linear-gradient(135deg, var(--color-primary), var(--color-primary-dark));
        }}
        
        .category-political {{
            background: linear-gradient(135deg, #9b59b6, #8e44ad);
        }}
        
        .category-social {{
            background: linear-gradient(135deg, var(--color-warning), #f39c12);
        }}
        
        .category-ethical {{
            background: linear-gradient(135deg, var(--color-danger), #c0392b);
        }}
        
        .category-historical {{
            background: linear-gradient(135deg, #1abc9c, #16a085);
        }}
        
        /* Default style for any other categories */
        [class^="category-"] {{
            background: linear-gradient(135deg, #95a5a6, #7f8c8d);
        }}
        
        /* Perspective Gaps & Integrative Insights */
        .perspective-gaps, .integrative-insights {{
            background-color: rgba(255, 255, 255, 0.7);
            border-radius: 8px;
            padding: 15px 20px;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }}
        
        .perspective-gaps {{
            border-left: 5px solid var(--color-info);
        }}
        
        .integrative-insights {{
            border-left: 5px solid #6f42c1;
        }}
        
        /* Viewpoint section styling */
        .viewpoint-section {{
            margin-bottom: 25px;
        }}
        
        .viewpoint-section h3 {{
            display: flex;
            align-items: center;
            margin-bottom: 15px;
            color: var(--color-heading);
        }}
        
        .section-icon {{
            display: inline-block;
            margin-right: 8px;
            font-size: 1.2em;
        }}
        
        /* Blockquote styling */
        blockquote {{
            border-left: 3px solid var(--color-primary);
            background-color: var(--color-bg-secondary);
            padding: 10px 20px;
            margin: 15px 0;
            font-style: italic;
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
        <div id="introduction" class="dr-section">
            <h2>Introduction</h2>
            {introduction_html}
        </div>
        
        <!-- Sub-Question Sections -->
        {sub_questions_html}
        
        <!-- Viewpoint Analysis Section (if available) -->
        {viewpoint_analysis_html}
        
        <!-- Conclusion -->
        <div id="conclusion" class="dr-section">
            <h2>Conclusion</h2>
            {conclusion_html}
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
<div id="question-{index}" class="dr-section">
    <div class="question-header">
        <h2>{index}. {question}</h2>
        <span class="dr-confidence dr-confidence--{confidence}">
            <span class="confidence-icon">
                {confidence_icon}
            </span>
            Confidence: {confidence_upper}
        </span>
    </div>
    
    <div class="content">
        <p>{answer}</p>
    </div>
    
    {info_gaps_html}
    
    {key_sources_html}
</div>
"""

# Template for viewpoint analysis section in the static HTML report
VIEWPOINT_ANALYSIS_TEMPLATE = """
<div id="viewpoint-analysis" class="dr-section viewpoint-analysis">
    <h2>Viewpoint Analysis</h2>
    
    <div class="viewpoint-section">
        <h3><span class="section-icon">🤝</span> Points of Agreement</h3>
        <div class="viewpoint-agreement">
            <ul>
                {agreements_html}
            </ul>
        </div>
    </div>
    
    <div class="viewpoint-section">
        <h3><span class="section-icon">⚖️</span> Areas of Tension</h3>
        <div class="viewpoint-tensions">
            {tensions_html}
        </div>
    </div>
    
    <div class="viewpoint-section">
        <h3><span class="section-icon">🔍</span> Perspective Gaps</h3>
        <div class="perspective-gaps">
            <p>{perspective_gaps}</p>
        </div>
    </div>
    
    <div class="viewpoint-section">
        <h3><span class="section-icon">💡</span> Integrative Insights</h3>
        <div class="integrative-insights">
            <p>{integrative_insights}</p>
        </div>
    </div>
</div>
"""

MCP_PROMPT = """This is the final stage in a multi-step research-pipeline.

You will be given the following information:

- the original user query
- written text synthesis that was generated based on the search data
- analysis data containing reflection and viewpoint analysis

You have the following tools available to you:

- web_search_exa: Real-time web search with content extraction
- research_paper_search: Academic paper and research content search
- company_research: Company website crawling for business information
- crawling: Extract content from specific URLs
- competitor_finder: Find company competitors
- linkedin_search: Search LinkedIn for companies and people
- wikipedia_search_exa: Retrieve information from Wikipedia articles
- github_search: Search GitHub repositories and issues

Please use the tools to search for anything you feel might still be needed to
answer or to round out the research. The results of what you find will be passed
to the final report generation and summarization step.

## User Query
<user_query>
{user_query}
</user_query>

## Synthesis Data

### Synthesized Info
<synthesized_info>
{synthesized_info}
</synthesized_info>

### Enhanced Info
<enhanced_info>
{enhanced_info}
</enhanced_info>

## Analysis Data

### Viewpoint Analysis
<viewpoint_analysis>
{viewpoint_analysis}
</viewpoint_analysis>

### Reflection Metadata
<reflection_metadata>
{reflection_metadata}
</reflection_metadata>

Now please use the tools to search for anything you feel might still be needed to
answer or to round out the research. The results of what you find will be passed
to the final report generation and summarization step.

Format your output as a well-structured conclusion section in HTML format with
appropriate paragraph breaks and formatting. Use <p> tags for paragraphs and
organize the content logically with clear transitions between the different
aspects outlined above.
"""
