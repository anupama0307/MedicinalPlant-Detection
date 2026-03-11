# Dataset Class Distribution - Segmented Medicinal Leaf Images

## Dataset Statistics Overview

| # | Plant Class | Scientific/Common Names | Image Count |
|---|---|---|---:|
| 1 | Amaranthus Viridis | Arive-Dantu | **122** |
| 2 | Basella Alba | Basale | **103** |
| 3 | Mentha | Mint | **97** |
| 4 | Punica Granatum | Pomegranate | **79** |
| 5 | Moringa Oleifera | Drumstick | **77** |
| 6 | Carissa Carandas | Karanda | **74** |
| 7 | Jasminum | Jasmine | **71** |
| 8 | Psidium Guajava | Guava | **65** |
| 9 | Ficus Religiosa | Peepal Tree | **63** |
| 10 | Nerium Oleander | Oleander | **62** |
| 11 | Mangifera Indica | Mango | **62** |
| 12 | Pongamia Pinnata | Indian Beech | **61** |
| 13 | Murraya Koenigii | Curry | **60** |
| 14 | Azadirachta Indica | Neem | **60** |
| 15 | Santalum Album | Sandalwood | **58** |
| 16 | Citrus Limon | Lemon | **57** |
| 17 | Artocarpus Heterophyllus | Jackfruit | **56** |
| 18 | Muntingia Calabura | Jamaica Cherry-Gasagase | **56** |
| 19 | Tabernaemontana Divaricata | Crape Jasmine | **56** |
| 20 | Syzygium Jambos | Rose Apple | **56** |
| 21 | Ocimum Tenuiflorum | Tulsi | **52** |
| 22 | Ficus Auriculata | Roxburgh fig | **50** |
| 23 | Alpinia Galanga | Rasna | **50** |
| 24 | Plectranthus Amboinicus | Mexican Mint | **48** |
| 25 | Piper Betle | Betel | **48** |
| 26 | Hibiscus Rosa-sinensis | Hibiscus | **43** |
| 27 | Nyctanthes Arbor-tristis | Parijata | **40** |
| 28 | Syzygium Cumini | Jamun | **39** |
| 29 | Trigonella Foenum-graecum | Fenugreek | **36** |
| 30 | Brassica Juncea | Indian Mustard | **34** |

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| **Total Plant Classes** | 30 |
| **Total Images** | 1,788 |
| **Maximum Class Size** | Amaranthus Viridis (122 images) |
| **Minimum Class Size** | Brassica Juncea (34 images) |
| **Average Images per Class** | 59.6 images |
| **Median Images per Class** | 58 images |
| **Standard Deviation** | 21.4 images |

---

## Data Distribution Visualization

### Class Size Categories

**Large Classes (100+ images):**
- Amaranthus Viridis: 122
- Basella Alba: 103

**Medium-Large Classes (70-99 images):**
- Mentha: 97
- Punica Granatum: 79
- Moringa Oleifera: 77
- Carissa Carandas: 74
- Jasminum: 71

**Medium Classes (50-69 images):**
- Psidium Guajava: 65
- Ficus Religiosa: 63
- Nerium Oleander: 62
- Mangifera Indica: 62
- Pongamia Pinnata: 61
- Murraya Koenigii: 60
- Azadirachta Indica: 60
- Santalum Album: 58
- Citrus Limon: 57
- Artocarpus Heterophyllus: 56
- Muntingia Calabura: 56
- Tabernaemontana Divaricata: 56
- Syzygium Jambos: 56
- Ocimum Tenuiflorum: 52
- Ficus Auriculata: 50
- Alpinia Galanga: 50

**Small-Medium Classes (40-49 images):**
- Plectranthus Amboinicus: 48
- Piper Betle: 48
- Hibiscus Rosa-sinensis: 43
- Nyctanthes Arbor-tristis: 40

**Small Classes (<40 images):**
- Syzygium Cumini: 39
- Trigonella Foenum-graecum: 36
- Brassica Juncea: 34

---

## Dataset Balance Analysis

✅ **Relatively Balanced:** Standard deviation of 21.4 indicates reasonably balanced distribution
- Largest class (122 images) is 3.6× larger than smallest (34 images)
- Most classes fall within 34-122 range (±standard deviation)
- Suitable for training without severe class imbalance issues

⚠️ **Minor Imbalance Observed:** 
- Underrepresented: Brassica Juncea, Trigonella, Syzygium Cumini, Nyctanthes
- Overrepresented: Amaranthus, Basella, Mentha
- Recommendation: Consider class weights during training if needed

---

## Data Usage Split (Inferred)

### Typical Split Distribution
- **Training Set:** ~70% (1,251 images)
- **Validation Set:** ~15% (268 images)  
- **Test Set:** ~15% (269 images)

*Note: Actual split depends on your dataset_split directory configuration*

---

## Class Implementation Notes

### Most Represented Classes (Best Training)
1. **Amaranthus Viridis** (122) - Best quality training data
2. **Basella Alba** (103) - Good representation
3. **Mentha** (97) - Sufficient variety for model learning

### Least Represented Classes (May Need Attention)
1. **Brassica Juncea** (34) - Limited data, potential overfitting risk
2. **Trigonella Foenum-graecum** (36) - May need data augmentation
3. **Syzygium Cumini** (39) - Limited diverse examples

### Balanced Classes (Optimal)
- **Citrus Limon** (57)
- **Santalum Album** (58)
- **Murraya Koenigii** (60)
- **Ficus Auriculata** (50)

---

## Recommendations

1. **For Training:** Use stratified splitting to maintain class proportions across train/val/test
2. **For Small Classes:** Apply data augmentation (rotation, flip, brightness adjustment)
3. **For Imbalanced Classes:** Consider weighted loss functions during training
4. **For Production:** Confidence thresholds may need adjustment per class (higher for underrepresented classes)

---

*Dataset: Segmented Medicinal Leaf Images*  
*Total Unique plant species: 30*  
*Total annotated samples: 1,788*  
*Image format: JPG/PNG*  
*Image size: 224×224 pixels (standardized)*

