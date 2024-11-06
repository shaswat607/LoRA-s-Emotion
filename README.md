# Enhancing AI's Emotional Understanding in Conversations Through PEFT Techniques and Custom Embeddings
`Welcome to LoRA's Emotion. `
#### Quick links:
- [My Blog on Medium about the research](https://medium.com/@shaswat607/advancing-ais-emotional-intelligence-in-conversations-exploring-peft-techniques-and-custom-c76f3ed94a67)
- [Setup instructions](#Setup)



This repository is about capturing the subtle art of human emotion through the lens of artificial intelligence. It explores the intricate dance between words and feelings, employing advanced NLP techniques and fine-tuning methods to craft a model that perceives and responds with the empathy and depth of a true conversationalist. Here lies the fusion of cognitive science and machine learning, a step towards building AI that resonates with the human spirit.

#### This research emphasizes resource and time-effectiveness while addressing the challenge

 - `Small LLM`
 - `Approx 10 minutes of training`
 - `Single RTX 4090 GPU`


## How This Research Was Conducted: A Detailed Explanation

In a quest to bridge the gap between human emotion and artificial intelligence, this research delves deep into the intricate process of enabling AI to comprehend and respond to the complex emotional nuances in human conversations. By fine-tuning the FLAN T5 Base model using advanced techniques like Parameter-Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation), the goal was to craft a model that resonates with human-like empathy and understanding.

## 1. Selection of FLAN T5 Base Model

Despite the versatility of the latest NLP models with billions of parameters, deploying them for customized use cases is often impractical due to several factors:

- **Resource Intensity:** Training and fine-tuning these large models require substantial computational resources, making the process expensive and time-consuming.

- **Practical Constraints:** Their massive size can make deployment challenging or even impossible, especially for real-time applications or environments with limited infrastructure.

- **Diminishing Returns:** For specialized tasks, the marginal performance gains may not justify the exponential increase in resource requirements.

### Harnessing the Power of Smaller Models

To navigate these challenges, it's often more pragmatic to select the smallest model possible that still embodies the capabilities required for the task at hand. This approach ensures efficiency and feasibility without compromising on performance. Hence, the FLAN T5 Base model was chosen for this research.

### Why FLAN T5 Base?
The FLAN T5 Base is a sequence-to-sequence transformer model featuring both an encoder and a decoder. This architecture was chosen because the encoder effectively captures the emotional context of input conversations, while the decoder, using an auto-regressive approach, generates responses that maintain appropriate emotional alignment. This combination ensures that the model not only understands the nuanced emotions conveyed but also responds in a way that appropriately navigates the emotional landscape of the dialogue.


## 2. Parameter-Efficient Fine-Tuning (PEFT) with LoRA
Traditional fine-tuning can be resource-intensive, especially for large models. To address this, Parameter-Efficient Fine-Tuning (PEFT) using LoRA (Low-Rank Adaptation) was implemented. LoRA allows fine-tuning by introducing new parameters while retaining the pre-trained weights frozen. This significantly reduces the computational overhead and enhances the model’s adaptability to specific tasks without requiring full-scale fine-tuning.

Implementation Details:

 - **Layer Injection:** LoRA layers were embedded into the transformer structure, targeting attention and feed-forward submodules.

- **Low-Rank Updates:** Training focused on learning low-rank updates to adapt the model to the emotional context of conversations efficiently.

- **Training Strategies:** Carefully selected training parameters such as learning rate, batch size, and evaluation intervals were used to optimize performance.

## 3. Data Collection and Preparation
I generated and utilized synthetic data for training, consisting of simulated conversations between individuals and therapists that mirror how therapists understand and navigate human emotions. The intent was to develop a model that can emulate a therapist's way of responding to a person's emotional journey, providing empathetic support as an AI-powered therapeutic assistant.

### Insight about versatility
The synthetic data was generated to capture vastness of therapeutic consultations with **15 Unique Therapies:**


- Acceptance and Commitment Therapy (ACT)
- Behavioral Therapy
- Behavioral Activation
- Dialectical Behavior Therapy (DBT)
- Mindfulness-Based Stress Reduction (MBSR)
- Cognitive Behavioral Therapy (CBT)
- Solution-Focused Brief Therapy (SFBT)
- Positive Psychology
- Motivational Interviewing (MI)
- Grief Counseling
- Interpersonal Therapy (IPT)
- Exposure Therapy
- Self-Compassion Therapy
- Mindfulness Techniques
- Mindfulness-Based Cognitive Therapy (MBCT)

**And Total 146 Unique Categories:**
*like*
- Coping with Burnout
- Life Transitions
- Coping with Self-Isolation
- Work-Life Balance
- Decision-Making and Self-Trust
- Overcoming Loneliness
- Dealing with Decision Paralysis
- Building Resilience to Stress
- Strengthening Self-Compassion
- Managing Social Media Overload
- Gaining Motivation for Physical Activity
- Navigating Career Changes
- Reducing Procrastination
- Improving Emotional Resilience
- Improving Patience in Stressful Situations
- Enhancing Patience and Self-Discipline
- Mindful Focus on the Present, *etc*..

### Schema:
- **`conversations`**: List of conversations containing structured exchanges.
  - **`category`**: The thematic category of the conversation (e.g., "Career Growth and Self-Esteem", "Relationship Challenges and Boundaries").
  - **`therapy`**: The type of therapy approach applied in the conversation (e.g., "Cognitive Behavioral Therapy (CBT)", "Interpersonal Therapy (IPT)").
  - **`conversation`**: A list of interaction turns between the user and chatbot.
    - **`turn`**: The sequence number of the turn in the conversation.
    - **`speaker`**: Identifies whether the speaker is the "User" or "Chatbot".
    - **`text`**: The content of the speaker's message.
    - **`emotion_label`**: A list of emotions associated with the speaker's message (e.g., "Self-Doubt", "Frustration", "Supportive").

### Example Entries:
#### 1. Category: Career Growth and Self-Esteem
- **Therapy**: Cognitive Behavioral Therapy (CBT)
- **Sample Interaction**:
  - **Turn 1**: *User*: "I’m really struggling to believe in my skills, even though I’ve been told I’m good at my job." *(Emotions: Self-Doubt, Frustration)*
  - **Turn 2**: *Chatbot*: "It’s natural to feel uncertain at times. Have you tried recalling specific accomplishments to challenge those doubts?" *(Emotions: Encouraging)*
  - **Turn 3**: *User*: "I guess I can think of a few, but they don’t feel enough to silence my insecurities." *(Emotions: Insecurity)*
  - **Turn 4**: *Chatbot*: "Even small accomplishments are valuable. Sometimes acknowledging them helps build a foundation of confidence." *(Emotions: Supportive)*

#### 2. Category: Relationship Challenges and Boundaries
- **Therapy**: Interpersonal Therapy (IPT)
- **Sample Interaction**:
  - **Turn 1**: *User*: "I feel like my friend isn’t respecting my boundaries lately." *(Emotions: Frustration, Disappointment)*
  - **Turn 2**: *Chatbot*: "That sounds challenging. Setting boundaries with friends can be tough but very necessary." *(Emotions: Understanding)*
  - **Turn 3**: *User*: "I don’t want to hurt their feelings, though." *(Emotions: Anxiety, Concern)*
  - **Turn 4**: *Chatbot*: "It’s okay to be considerate, but remember, a true friend will understand your need for space." *(Emotions: Reassuring)*


## Custom Embeddings - Enhancing Emotional Empathy 
Custom encoding was essential to elevate the chatbot's ability to respond with genuine emotional empathy—a feature not inherently supported by the base google/flan-t5-base model. The standard T5 model processes text without explicit consideration of emotional context, leading to responses that can be generic or lacking in emotional depth. To address this limitation, it was necessary to implement a tailored encoding mechanism that allows the model to recognize and incorporate emotional cues from user inputs, thereby enabling more nuanced and empathetic interactions.

As part of this, I added an emotion embedding layer that maps specific emotional labels (e.g., Anxiety, Hope) to dense vector representations. Further, projected these emotion embeddings to match the model's d_model dimension and incorporated them directly into the input embeddings fed to the encoder. This allows the model to consider the emotional tone of each interaction.

## Results & Evaluation 
### Metric: BERTScore
We used BERTScore to evaluate our chatbot’s responses due to its ability to measure semantic similarity by leveraging pre-trained contextual embeddings. Unlike traditional metrics like BLEU or ROUGE, BERTScore captures the context and meaning of words, which is crucial for assessing conversational and empathetic responses.

### Results Overview
```
Average Precision: 0.7327

Average Recall: 0.7486

Average F1 Score: 0.7395
```

These scores indicate that our chatbot effectively generates contextually accurate and semantically meaningful responses. The high recall suggests comprehensive responses, while the balanced F1 score reflects overall robustness in maintaining relevance and accuracy. This confirms that the chatbot aligns well with human-like interaction, making it suitable for applications requiring supportive and empathetic communication.

Our results were based on synthetic data, which, while valuable for initial testing, may not fully represent the complexity of real-world conversations. Additionally, the size of the dataset plays a critical role in model performance. A larger dataset would provide more examples and context, improving the model's ability to generalize and generate more accurate, contextually nuanced responses. More data, especially real and diverse data, would likely lead to higher BERTScore results and better overall performance, making the chatbot more effective for real-world applications.

# Setup 
Ensure you have **Python 3.9.7** and **pip 24.2** installed. Follow the steps below to set up your environment using **Homebrew** and **pyenv**.

### Step 1: Install Python and pyenv
```bash
brew install python  
brew install pyenv
```

### Step 2: Configure pyenv
Add the following lines to your shell configuration file (e.g., `.bash_profile`, `.bashrc`, or `.zshrc`):

```bash
# Add pyenv to the shell
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
```

**Verify Configuration:**
```bash
cat ~/.bash_profile   # Use ~/.zshrc for Zsh users
```

**Apply Changes:**
```bash
source ~/.bash_profile   # Or source ~/.zshrc for Zsh users
```

### Step 3: Install and Set Python Version
```bash
pyenv install 3.9.7
pyenv global 3.9.7
```

**Confirm Installation:**
```bash
python3 --version  # Should output Python 3.9.7
```

## Install Project Dependencies

Ensure `pip` is up-to-date and install the required packages:

```bash
pip install --upgrade pip
pip install -U datasets==2.17.0
pip install --disable-pip-version-check torch==1.13.1 torchdata==0.5.1 --quiet
pip install transformers==4.27.2 evaluate==0.4.0 rouge_score==0.1.2 loralib==0.1.1 peft==0.3.0 --quiet
pip install scikit-learn
```

---

