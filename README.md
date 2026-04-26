# TensorFlow Transform + tf.data Pipeline Demo

A complete, production-ready demonstration of combining **tf.transform** and **tf.data** for scalable machine learning pipelines.

## 🎯 What This Demo Shows

```
Raw CSV Data 
    ↓
tf.transform (Apache Beam) - Feature Engineering
    ↓ (learns: vocab, mean, std, buckets)
Sharded TFRecord Files
    ↓
tf.data Pipeline - Efficient Loading
    ↓ (parallel: interleave, prefetch, AUTOTUNE)
Model Training
    ↓
Inference - with Automatic Transform Consistency
```
## Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STEP 1: RAW DATA                            │
│                                                                     │
│  generate_sample_data()                                            │
│         ↓                                                          │
│  CSV File: transactions.csv                                        │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │ transaction_id | product_category | price | quantity ... │     │
│  │ 1             | Electronics      | 299.99 | 2        ... │     │
│  │ 2             | Clothing         | 49.99  | 1        ... │     │
│  └──────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│              STEP 2: tf.transform (Apache Beam)                     │
│                                                                     │
│  Apache Beam Pipeline (DirectRunner, 2 workers)                    │
│         ↓                                                          │
│  PHASE 1: ANALYZE - Compute Statistics from ENTIRE Dataset         │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ • Scan all 1000 records                              │         │
│  │ • Compute vocabularies:                              │         │
│  │   - product_category → ['Electronics', 'Clothing'...] │         │
│  │   - customer_city → ['New York', 'Chicago'...]       │         │
│  │                                                      │         │
│  │ • Compute statistics:                                │         │
│  │   - price_mean = 254.73                             │         │
│  │   - price_std = 142.18                              │         │
│  │   - age_mean = 43.8                                 │         │
│  │   - age_std = 15.2                                  │         │
│  │                                                      │         │
│  │ • Compute quantile boundaries:                       │         │
│  │   - price_buckets = [10, 100, 200, 350, 500]        │         │
│  │   - age_buckets = [18, 30, 45, 60, 70]              │         │
│  └──────────────────────────────────────────────────────┘         │
│         ↓                                                          │
│  SAVE Transform Function (/data/transformed/)                      │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ transform_fn/                                        │         │
│  │ ├── saved_model.pb (transformation graph)           │         │
│  │ ├── assets/                                          │         │
│  │ │   ├── product_category_vocab (learned vocab)      │         │
│  │ │   └── customer_city_vocab (learned vocab)         │         │
│  │ └── variables/ (learned statistics)                 │         │
│  └──────────────────────────────────────────────────────┘         │
│         ↓                                                          │
│  PHASE 2: TRANSFORM - Apply Learned Transformations                │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ For each record:                                     │         │
│  │ • 'Electronics' → 0 (using learned vocab)           │         │
│  │ • price 299.99 → 0.318 (using learned μ, σ)        │         │
│  │ • age 35 → -0.579 (using learned μ, σ)             │         │
│  │ • price bucket → 2 (using learned boundaries)       │         │
│  └──────────────────────────────────────────────────────┘         │
│         ↓                                                          │
│  WRITE to Sharded TFRecords (4 shards)                            │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                 STEP 3: SHARDED TFRECORD FILES                      │
│                                                                     │
│  /data/tfrecords/                                                  │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ train-00000-of-00004.tfrecord (~250 records)        │         │
│  │ train-00001-of-00004.tfrecord (~250 records)        │         │
│  │ train-00002-of-00004.tfrecord (~250 records)        │         │
│  │ train-00003-of-00004.tfrecord (~250 records)        │         │
│  └──────────────────────────────────────────────────────┘         │
│                                                                     │
│  Benefits:                                                          │
│  • Binary format (efficient storage & I/O)                         │
│  • Pre-transformed (no runtime transformation cost)                │
│  • Sharded (enables parallel reading)                              │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│          STEP 4: tf.data Pipeline (Parallel Loading)               │
│                                                                     │
│  list_files() - Shuffle file order                                 │
│         ↓                                                          │
│  interleave(cycle_length=AUTOTUNE) - Read 4 files in parallel      │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ Thread 1: Read from train-00000-of-00004.tfrecord   │         │
│  │ Thread 2: Read from train-00001-of-00004.tfrecord   │ Parallel │
│  │ Thread 3: Read from train-00002-of-00004.tfrecord   │ Reading  │
│  │ Thread 4: Read from train-00003-of-00004.tfrecord   │         │
│  └──────────────────────────────────────────────────────┘         │
│         ↓                                                          │
│  map(parse, num_parallel_calls=AUTOTUNE) - Parse in parallel       │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ Worker 1: Parse TFRecord → Dict                      │         │
│  │ Worker 2: Parse TFRecord → Dict                      │ Parallel │
│  │ Worker 3: Parse TFRecord → Dict                      │ Parsing  │
│  │ Worker 4: Parse TFRecord → Dict                      │         │
│  └──────────────────────────────────────────────────────┘         │
│         ↓                                                          │
│  shuffle(buffer_size=1000) - Shuffle records                       │
│         ↓                                                          │
│  batch(32) - Create batches                                        │
│         ↓                                                          │
│  prefetch(buffer_size=AUTOTUNE) - Overlap with training            │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ While GPU trains batch N:                            │         │
│  │   CPU prepares batch N+1 in parallel                 │         │
│  │                                                      │         │
│  │ No idle time! Continuous pipeline.                   │         │
│  └──────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 5: MODEL TRAINING                           │
│                                                                     │
│  model.fit(train_dataset, epochs=5)                                │
│         ↓                                                          │
│  For each batch from tf.data:                                      │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ 1. Forward pass                                      │         │
│  │ 2. Compute loss                                      │         │
│  │ 3. Backpropagation                                   │         │
│  │ 4. Update weights                                    │         │
│  └──────────────────────────────────────────────────────┘         │
│         ↓                                                          │
│  Save trained model (/data/model/)                                 │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│          STEP 6: INFERENCE (Transform Consistency!)                 │
│                                                                     │
│  New raw input from user:                                          │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ product_category: 'Electronics'                      │         │
│  │ price: 299.99                                        │         │
│  │ customer_city: 'Seattle'                             │         │
│  │ customer_age: 35                                     │         │
│  └──────────────────────────────────────────────────────┘         │
│         ↓                                                          │
│  Load SAVED transform function                                     │
│  tf_transform_output = tft.TFTransformOutput(transform_dir)        │
│         ↓                                                          │
│  Apply SAME transformations (using SAVED statistics)               │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ • 'Electronics' → 0 (using SAVED vocab)             │         │
│  │ • 299.99 → 0.318 (using SAVED μ=254.73, σ=142.18)  │         │
│  │ • 'Seattle' → 8 (using SAVED vocab)                 │         │
│  │ • 35 → -0.579 (using SAVED μ=43.8, σ=15.2)         │         │
│  │ • price bucket → 2 (using SAVED boundaries)         │         │
│  └──────────────────────────────────────────────────────┘         │
│         ↓                                                          │
│  Feed to model → Get prediction                                    │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ Prediction: 0.6234                                   │         │
│  │ Purchase probability: 62.34%                         │         │
│  └──────────────────────────────────────────────────────┘         │
│                                                                     │
│  ✓ SAME transformations as training (automatic consistency!)       │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Parallelization Points

### 1. Apache Beam (tf.transform)
- **2 workers** process different portions of the dataset
- **ANALYZE phase**: Workers scan different partitions in parallel
- **TRANSFORM phase**: Workers transform different records in parallel

### 2. tf.data Pipeline
- **interleave(cycle_length=AUTOTUNE)**: Reads multiple TFRecord shards simultaneously
- **map(num_parallel_calls=AUTOTUNE)**: Parses multiple records in parallel
- **prefetch(buffer_size=AUTOTUNE)**: Prepares next batch while current batch trains

### 3. Why AUTOTUNE?
TensorFlow dynamically adjusts based on:
- Number of CPU cores
- Memory availability
- Data loading bottlenecks
- GPU/CPU synchronization

## Transformation Consistency Example

### Training Time:
```python
# tf.transform computes from 1000 records:
price_mean = 254.73
price_std = 142.18

# Saves to transform_fn/variables/
```

### Serving Time:
```python
# Loads SAME statistics from transform_fn/
price_mean = 254.73  # Automatically loaded
price_std = 142.18   # Automatically loaded

# Applies SAME transformation:
normalized = (299.99 - 254.73) / 142.18 = 0.318
```

**No manual code duplication! No training-serving skew!**

## Performance Comparison

### Without Parallelization (Sequential):
```
Time to read 4 shards: 4 * 100ms = 400ms
Time to parse 1000 records: 1000 * 1ms = 1000ms
Total: 1400ms per epoch
```

### With Parallelization (This Demo):
```
Time to read 4 shards in parallel: max(100ms) = 100ms
Time to parse in parallel (4 workers): 1000ms / 4 = 250ms
Overlapped with training (prefetch): ~0ms visible latency
Total: ~350ms per epoch (4x speedup!)
```

## File Size Comparison

### CSV (Raw):
- Size: ~150 KB
- Format: Text
- Read speed: Moderate
- Parse overhead: High (string parsing)

### TFRecord (Transformed):
- Size: ~80 KB (compressed binary)
- Format: Protocol Buffer
- Read speed: Fast (sequential binary)
- Parse overhead: Low (binary deserialization)
- **Bonus**: Pre-transformed (no runtime transformation cost)

## When to Use What?

### Use tf.transform when:
- ✅ Dataset > 100K records
- ✅ Complex feature engineering
- ✅ Production deployment
- ✅ Need transformation consistency
- ✅ Scaling to distributed systems

### Use tf.data always when:
- ✅ Training TensorFlow models
- ✅ Need efficient data loading
- ✅ Want automatic parallelization
- ✅ Working with large datasets

### Scale to Cloud when:
- ✅ Dataset > 1M records
- ✅ Need distributed processing
- ✅ Have budget for cloud resources

```python
# Just change Apache Beam runner:
pipeline_options = PipelineOptions(
    runner='DataflowRunner',  # Google Cloud Dataflow
    project='my-gcp-project',
    temp_location='gs://my-bucket/temp',
    region='us-central1'
)
# Everything else stays the same!
```

This pattern scales from laptop → single machine → cloud cluster seamlessly.


## ✨ Key Features

### tf.transform Capabilities Demonstrated:
- ✅ **Vocabulary Learning**: Categorical feature encoding
- ✅ **Mean & Standard Deviation**: Numerical normalization
- ✅ **Bucketization Boundary Estimation**: Quantile-based bucketing
- ✅ **Apache Beam Integration**: Parallel processing for large datasets
- ✅ **Transform Consistency**: Same transformations in training and serving

### tf.data Parallelization Demonstrated:
- ✅ **Interleave**: Parallel file reading with `cycle_length=AUTOTUNE`
- ✅ **Parallel Calls**: `num_parallel_calls=AUTOTUNE`
- ✅ **Prefetching**: `prefetch(buffer_size=AUTOTUNE)`
- ✅ **TFRecordDataset**: Efficient binary format
- ✅ **Sharded Files**: Distributed data loading

## 📋 Prerequisites

- check notebook 1st cell

### Step 2: Run the Demo

The script will:
1. Generate sample CSV data (1000 records)
2. Run tf.transform with Apache Beam
3. Create sharded TFRecord files
4. Build an efficient tf.data pipeline
5. Train a simple model
6. Demonstrate serving with transform consistency

## 📊 Generated Data

The demo generates synthetic e-commerce transaction data:

| Column | Type | Description |
|--------|------|-------------|
| transaction_id | int | Unique transaction ID |
| product_category | string | Electronics, Clothing, Books, Home, Sports |
| price | float | Product price ($10-$500) |
| quantity | int | Quantity purchased (1-10) |
| customer_city | string | 10 major US cities |
| customer_age | int | Customer age (18-70) |
| discount_applied | int | Binary (0 or 1) |
| target_purchased | int | Binary target variable |

**No external download required** - data is generated automatically!

## 🔧 What tf.transform Learns

During the `ANALYZE` phase, tf.transform computes and saves:

### 1. Vocabularies (Categorical Features)
- **product_category**: Maps ['Electronics', 'Clothing', ...] → [0, 1, 2, ...]
- **customer_city**: Maps ['New York', 'Los Angeles', ...] → [0, 1, 2, ...]

### 2. Statistics (Numerical Features)
- **price**: Mean = μ, Std Dev = σ
- **customer_age**: Mean = μ, Std Dev = σ
- **total_amount**: Mean = μ, Std Dev = σ

### 3. Bucketization Boundaries
- **price_bucketized**: 5 quantile-based buckets
- **age_bucketized**: 4 age group buckets

These statistics are saved and **automatically reused during serving**!

## 🎯 Pipeline Architecture

### Phase 1: tf.transform with Apache Beam

```python
Raw CSV → Read → Parse → ANALYZE (compute stats) → TRANSFORM → Write TFRecords
                                    ↓
                            Save Transform Function
```

**Apache Beam Configuration:**
- Runner: DirectRunner (local execution)
- Workers: 2 (parallel processing)
- Mode: multi_processing

### Phase 2: tf.data Pipeline

```python
TFRecord Shards → list_files (shuffle)
                      ↓
              interleave (cycle_length=AUTOTUNE)
                      ↓
              map parse (num_parallel_calls=AUTOTUNE)
                      ↓
              shuffle → batch → prefetch (AUTOTUNE)
```

**Parallelization Strategy:**
- **Interleave**: Reads from multiple files simultaneously
- **Parallel Calls**: Parses multiple records in parallel
- **Prefetch**: Overlaps data loading with model training
- **AUTOTUNE**: TensorFlow dynamically optimizes parallelism

## 🔑 Key Insights

### 1. Transformation Consistency (Training vs Serving)

**Traditional Approach (Error-Prone):**
```python
# Training
train_mean = df['price'].mean()
train_std = df['price'].std()
df['price_norm'] = (df['price'] - train_mean) / train_std

# Serving - MANUAL DUPLICATION (can go wrong!)
serving_price_norm = (new_price - train_mean) / train_std
```

**tf.transform Approach (Automatic Consistency):**
```python
# Training - learns and saves statistics
def preprocessing_fn(inputs):
    mean = tft.mean(inputs['price'])  # Computed once
    std = tft.var(inputs['price']) ** 0.5
    return (inputs['price'] - mean) / std

# Serving - uses SAME saved statistics automatically
transform_fn = tft.TFTransformOutput(transform_dir)
transformed = transform_fn.transform_raw_features(raw_input)
# Statistics are loaded automatically - no manual duplication!
```

### 2. Why Sharded TFRecords?

- **Parallel Reading**: Multiple workers can read different shards
- **Efficient Storage**: Binary format, smaller than CSV
- **Fast I/O**: Sequential reads, optimal for training
- **Scalability**: Easy to distribute across machines

### 3. tf.data AUTOTUNE

TensorFlow automatically tunes these parameters based on:
- CPU/GPU availability
- Data loading bottlenecks
- Memory constraints

You don't need to manually set `cycle_length=4` or `num_parallel_calls=8`. Just use `tf.data.AUTOTUNE`!

## 📁 Output Structure

After running, you'll have:

```
/home/data/
├── raw/
│   └── transactions.csv          # Generated raw data
├── transformed/
│   ├── transform_fn/             # Saved transform function
│   │   ├── saved_model.pb
│   │   ├── assets/
│   │   │   ├── product_category_vocab
│   │   │   └── customer_city_vocab
│   │   └── variables/
│   └── tmp/                      # Beam temp files
├── tfrecords/
│   ├── train-00000-of-00004.tfrecord
│   ├── train-00001-of-00004.tfrecord
│   ├── train-00002-of-00004.tfrecord
│   └── train-00003-of-00004.tfrecord
└── model/
    └── simple_model/             # Trained Keras model
```

## 🔍 Inspecting Results

### View Learned Vocabularies

```python
import tensorflow_transform as tft

tf_transform_output = tft.TFTransformOutput('/home/data/transformed')

# Read vocabularies
with open('/home/data/transformed/transform_fn/assets/product_category_vocab', 'r') as f:
    print("Product Categories:", f.read().splitlines())

with open('/home/data/transformed/transform_fn/assets/customer_city_vocab', 'r') as f:
    print("Cities:", f.read().splitlines())
```

### View Transform Metadata

```python
# Get feature statistics
transform_output = tft.TFTransformOutput('/home/data/transformed')
print(transform_output.transformed_feature_spec())
```

## 🎓 Learning Points

### 1. When to Use tf.transform?
- Large datasets (>1M records)
- Complex feature engineering
- Need transformation consistency
- Building production ML systems

### 2. When to Use tf.data?
- Always! It's the recommended way to load data in TensorFlow
- Especially important for large datasets
- Enables parallel and distributed training

### 3. Apache Beam vs Direct Execution
- **DirectRunner** (this demo): Local execution, good for prototyping
- **DataflowRunner**: Google Cloud, scales to billions of records
- **SparkRunner**: Apache Spark clusters

## 🔧 Customization

### Change Number of Shards

```python
NUM_SHARDS = 8  # Increase for larger datasets
```

### Adjust Batch Size

```python
BATCH_SIZE = 64  # Larger batches for GPU training
```

### Modify Features

Edit `preprocessing_fn()` to add/remove transformations:

```python
def preprocessing_fn(inputs):
    outputs = {}
    
    # Add your transformations
    outputs['new_feature'] = tft.scale_to_z_score(inputs['column_name'])
    
    return outputs
```

## 📈 Performance Tips

### For Larger Datasets

1. **Increase Apache Beam Workers:**
```python
pipeline_options = PipelineOptions(
    direct_num_workers=4,  # More workers
)
```

2. **More TFRecord Shards:**
```python
NUM_SHARDS = 16  # Better parallelism
```

3. **Larger Shuffle Buffer:**
```python
dataset = dataset.shuffle(buffer_size=10000)
```

### For GPU Training

1. **Larger Batch Size:**
```python
BATCH_SIZE = 128
```

2. **Mixed Precision:**
```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

## 🐛 Troubleshooting

### Out of Memory
- Reduce `NUM_SHARDS`
- Reduce `direct_num_workers`
- Use smaller dataset

### Slow Execution
- Increase `NUM_SHARDS` for better parallelism
- Check if prefetch buffer is too small
- Verify AUTOTUNE is being used

## ⭐ Key Takeaways

1. **tf.transform eliminates training-serving skew** by saving transformation logic
2. **Apache Beam enables scalable preprocessing** from laptops to cloud clusters
3. **tf.data AUTOTUNE handles parallelization** automatically
4. **Sharded TFRecords enable efficient distributed training**
5. **This pattern scales** from prototypes to production systems

---


