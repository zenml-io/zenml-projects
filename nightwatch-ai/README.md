# 🌙 NightWatch: AI Database Summaries While You Sleep

> Wake up to AI-generated insights from your Supabase database every morning. This ZenML pipeline uses OpenAI's GPT-4 to analyze yesterday's database activity, compare it to historical trends, and deliver concise summaries directly to your Slack channels.

![The summary in Slack](assets/youtldr_summarizer_slack.png)

## ✨ Why NightWatch?

**NightWatch** transforms raw database activity into actionable business intelligence while you sleep. Instead of manually querying and analyzing your Supabase database each morning, let AI do the heavy lifting:

- 🔍 **Automated Analysis**: Daily summaries of your database activity delivered to Slack
- 📊 **Trend Detection**: Compare today's insights with historical patterns
- 🧠 **AI-Powered**: Leverage OpenAI's GPT-4 to extract meaningful insights
- 🔄 **Version Control**: Track changes over time with ZenML's data versioning
- ⚙️ **Customizable**: Adapt to your specific database schema and business needs
- 🚀 **Production-Ready**: Scale from local development to enterprise deployment

## 🎯 Use Cases

### Customer Support Intelligence
Transform support tickets and customer feedback into actionable insights. Identify common pain points, track sentiment trends, and prioritize product improvements based on real user feedback.

### E-commerce Analytics
Monitor product performance, track inventory movements, and identify purchasing patterns. Get daily summaries of sales trends, popular products, and inventory alerts.

### Content Platform Engagement
Understand what content resonates with your audience. NightWatch can analyze user engagement data to identify trending topics, popular creators, and content performance patterns.

### Application Performance Monitoring
Track user behavior, error rates, and performance metrics. Receive daily summaries highlighting potential issues before they become critical problems.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Access to a Supabase database
- OpenAI API key with GPT-4 access
- ZenML account (free tier available)

### Quick Installation

1. **Install required packages**:
   ```bash
   pip install -r src/requirements.txt
   ```

2. **Connect to your ZenML deployment**:
   ```bash
   zenml login https://your-zenml-instance.com
   ```

3. **Set up your secrets**:
   ```bash
   # Configure Supabase connection
   zenml secret create supabase \
       --supabase_url=$SUPABASE_URL \
       --supabase_key=$SUPABASE_KEY

   # Configure OpenAI access
   zenml secret create openai --api_key=$OPENAPI_API_KEY   
   ```

4. **Run your first summary**:
   ```bash
   python run.py
   ```

## 🛠️ Customization Options

### Custom Database Queries
Tailor the database queries to focus on the metrics that matter most to your business by modifying the [`importer` step](src/steps/importers.py):

```python
# Example: Focus on high-priority customer tickets
query = """
    SELECT * FROM support_tickets 
    WHERE created_at > NOW() - INTERVAL '24 hours'
    AND priority = 'high'
"""
```

### Personalized AI Prompts
Customize how the AI interprets your data by adjusting the prompts in the [`generate_summary` step](src/steps/summarizers.py):

```python
# Example: Focus on actionable insights
system_prompt = """
    Analyze the database activity and identify:
    1. Urgent issues requiring immediate attention
    2. Emerging trends compared to previous periods
    3. Recommended actions based on the data
"""
```

### Notification Channels
Configure where and how your summaries are delivered by customizing the alerter component.

## 🔧 Advanced Configuration

### Deploying to Production

NightWatch seamlessly scales from local development to production environments. Deploy on production-ready orchestrators:

1. **Install required integrations**:
   ```bash
   zenml integration install gcp slack -y
   ```

2. **Configure cloud storage**:
   ```bash
   zenml artifact-store register gcp_store -f gcp --path=gs://YOUR_BUCKET_PATH
   ```

3. **Set up Slack notifications**:
   ```bash
   zenml alerter register slack_alerter -f slack \
       --slack_token=YOUR_SLACK_TOKEN \
       --default_slack_channel_id=YOUR_CHANNEL_ID
   ```

4. **Register your production stack**:
   ```bash
   zenml stack register -a gcp_store -o default --alerter=slack_alerter --active
   ```

### Automated Daily Execution

Set up GitHub Actions to run NightWatch automatically every day:

1. Store your secrets in your GitHub repository
2. Create a workflow file (`.github/workflows/nightwatch.yml`):

```yaml
name: NightWatch Daily Summary

on:
  schedule:
    - cron: '0 5 * * *'  # Run at 5 AM UTC daily

jobs:
  run_pipeline:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout repository
      uses: actions/checkout@v3
    
    # Additional configuration steps...
    
    - name: Run NightWatch
      run: python run.py
```

## 📈 Scaling Your Insights

NightWatch is built on ZenML, giving you access to a complete MLOps ecosystem:

- **Orchestration**: Scale with [Airflow](https://docs.zenml.io/stack-components/orchestrators/airflow) or [Kubeflow](https://docs.zenml.io/stack-components/orchestrators/kubeflow)
- **Storage**: Store artifacts on [cloud storage](https://docs.zenml.io/user-guides/starter-guide/cache-previous-executions)
- **Tracking**: Monitor experiments with [MLflow integration](https://docs.zenml.io/stack-components/experiment-trackers/mlflow)
- **Alerting**: Customize notifications through various channels

## 🔒 Security & Compliance

NightWatch handles sensitive database information with care:

- All credentials are stored securely using ZenML's secret management
- Data processing occurs within your infrastructure
- No sensitive data is shared with external services except for the specific prompts sent to OpenAI

## 🌟 Start Transforming Your Data Today

Stop manually analyzing database logs and start your day with AI-powered insights that drive better business decisions.

Ready to wake up to smarter insights? Get started with NightWatch today!

---

*NightWatch is powered by [ZenML](https://zenml.io), [OpenAI](https://openai.com/gpt4), and [Supabase](https://supabase.com).*