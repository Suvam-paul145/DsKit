"""
Demo: NLP Utilities
==================
This demo showcases NLP/text processing functions in dskit.
"""

from dskit import (
    basic_text_stats, advanced_text_clean,
    extract_text_features, sentiment_analysis
)
import pandas as pd

def create_sample_text_data():
    """Create sample text dataset"""
    return pd.DataFrame({
        'review': [
            'This product is AMAZING!!! I love it so much <3',
            'Terrible quality. Very disappointed :(',
            'Good value for money. Would recommend.',
            'DO NOT BUY! Complete waste of money!!!',
            'Pretty decent, works as expected.',
            'Best purchase ever! Highly recommended :)',
            'Not worth the price. Poor quality.',
            'Absolutely FANTASTIC! Five stars ⭐⭐⭐⭐⭐',
            'Average product, nothing special.',
            'Love it! Works perfectly 😊'
        ],
        'comment': [
            'Fast shipping, great service!',
            'Took forever to arrive.',
            'Customer service was helpful.',
            'Would not buy again.',
            'Meets my expectations.',
            'Exceeded expectations!',
            'Could be better.',
            'Amazing experience!',
            'It\'s okay, I guess.',
            'Very satisfied with purchase!'
        ]
    })


def demo_text_stats():
    """Demo 1: Basic text statistics"""
    print("=" * 60)
    print("DEMO 1: Basic Text Statistics")
    print("=" * 60)
    
    df = create_sample_text_data()
    
    print("\n📊 Sample text data:")
    print(df[['review']].head(3))
    
    print("\n🔍 Computing text statistics...")
    stats_df = basic_text_stats(df, text_cols=['review', 'comment'])
    
    print("\n✓ Text statistics:")
    print(stats_df.head())


def demo_text_cleaning():
    """Demo 2: Advanced text cleaning"""
    print("\n" + "=" * 60)
    print("DEMO 2: Advanced Text Cleaning")
    print("=" * 60)
    
    df = create_sample_text_data()
    
    print("\n📊 Original text samples:")
    print(df['review'].head(3).tolist())
    
    print("\n🧹 Cleaning text...")
    print("   - Converting to lowercase")
    print("   - Removing special characters")
    print("   - Removing extra spaces")
    print("   - Removing emojis")
    
    df_clean = advanced_text_clean(
        df, 
        text_cols=['review', 'comment'],
        remove_urls=True,
        remove_emails=True,
        remove_special_chars=True,
        remove_extra_spaces=True
    )
    
    print("\n✓ Cleaned text samples:")
    print(df_clean['review'].head(3).tolist())


def demo_text_features():
    """Demo 3: Extract text features"""
    print("\n" + "=" * 60)
    print("DEMO 3: Text Feature Extraction")
    print("=" * 60)
    
    df = create_sample_text_data()
    
    print(f"\n📊 Original columns: {list(df.columns)}")
    
    print("\n🔧 Extracting text features...")
    print("   - Character count")
    print("   - Word count")
    print("   - Average word length")
    print("   - Punctuation count")
    
    df_features = extract_text_features(df, text_cols=['review', 'comment'])
    
    print(f"\n✓ New columns: {list(df_features.columns)}")
    print("\n📊 Sample features:")
    feature_cols = [col for col in df_features.columns if 'review_' in col][:5]
    print(df_features[feature_cols].head())


def demo_sentiment_analysis():
    """Demo 4: Sentiment analysis"""
    print("\n" + "=" * 60)
    print("DEMO 4: Sentiment Analysis")
    print("=" * 60)
    
    df = create_sample_text_data()
    
    print("\n📊 Sample reviews:")
    for i, review in enumerate(df['review'].head(5), 1):
        print(f"  {i}. {review}")
    
    print("\n🔍 Analyzing sentiment...")
    try:
        df_sentiment = sentiment_analysis(df, text_cols=['review'])
        
        print("\n✓ Sentiment analysis completed:")
        print("\n📊 Results:")
        print(df_sentiment[['review', 'review_sentiment', 'review_sentiment_score']].head())
        
        print("\n📈 Sentiment distribution:")
        print(df_sentiment['review_sentiment'].value_counts())
        
    except ImportError:
        print("\n⚠️ Note: TextBlob not installed")
        print("   Install with: pip install textblob")
        print("   For demonstration, showing expected output format:")
        print("\n   Expected columns: review_sentiment, review_sentiment_score")
        print("   Sentiment values: positive, negative, neutral")
    except Exception as e:
        print(f"\n⚠️ Note: {str(e)}")


def demo_complete_nlp_pipeline():
    """Demo 5: Complete NLP pipeline"""
    print("\n" + "=" * 60)
    print("DEMO 5: Complete NLP Pipeline")
    print("=" * 60)
    
    df = create_sample_text_data()
    
    print(f"\n📊 Starting with {len(df)} text samples")
    
    print("\n🔧 Step 1: Computing text statistics...")
    df = basic_text_stats(df, text_cols=['review'])
    print(f"   ✓ Added basic statistics")
    
    print("\n🔧 Step 2: Cleaning text...")
    df = advanced_text_clean(df, text_cols=['review', 'comment'])
    print(f"   ✓ Text cleaned")
    
    print("\n🔧 Step 3: Extracting features...")
    df = extract_text_features(df, text_cols=['review'])
    print(f"   ✓ Features extracted")
    
    print("\n🔧 Step 4: Analyzing sentiment...")
    try:
        df = sentiment_analysis(df, text_cols=['review'])
        print(f"   ✓ Sentiment analyzed")
    except:
        print(f"   ⚠️ Sentiment analysis skipped (TextBlob not available)")
    
    print(f"\n✅ Pipeline completed! Final shape: {df.shape}")
    print(f"   Total columns: {len(df.columns)}")


if __name__ == "__main__":
    print("\n" + "📝" * 30)
    print("NLP UTILITIES DEMO".center(60))
    print("📝" * 30 + "\n")
    
    demo_text_stats()
    demo_text_cleaning()
    demo_text_features()
    demo_sentiment_analysis()
    demo_complete_nlp_pipeline()
    
    print("\n" + "✅" * 30)
    print("ALL DEMOS COMPLETED".center(60))
    print("✅" * 30 + "\n")
