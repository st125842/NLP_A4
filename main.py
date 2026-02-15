import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
import re
from bert import BERT
import torch.nn as nn
# ==========================================
# 1. SETUP & HELPER FUNCTIONS
# ==========================================
import pickle

# Load the dictionary
try:
    with open('word2id.pkl', 'rb') as f:
        word2id = pickle.load(f)
    print("word2id loaded!")
except FileNotFoundError:
    print("Error: word2id.pkl not found. You must train and save it first.")
# Ensure your model is in Eval mode
class SentenceBERT(nn.Module):
    def __init__(self, bert_model, embed_dim, num_classes=3):
        super(SentenceBERT, self).__init__()
        self.bert = bert_model
        
        # The Classifier Layer: Takes concatenated (u, v, |u-v|)
        # Input size is 3x the embedding dimension
        self.classifier = nn.Linear(embed_dim * 3, num_classes)
        self.device = bert_model.device

    def mean_pooling(self, token_embeddings, attention_mask):
        # token_embeddings shape: [batch_size, seq_len, embed_dim]
        # attention_mask shape: [batch_size, seq_len]
        
        # Mask out padding tokens (make them zero so they don't affect average)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum of all valid token vectors
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Count of valid tokens (avoid division by zero)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Average
        return sum_embeddings / sum_mask

    def forward(self, input_ids_a, input_ids_b):
        # 1. Create dummy segment_ids (All zeros for single sentences)
        # Your BERT expects segment_ids, but SBERT treats each sentence independently.
        segment_ids_a = torch.zeros_like(input_ids_a).to(self.device)
        segment_ids_b = torch.zeros_like(input_ids_b).to(self.device)

        # 2. Pass through YOUR BERT (Shared Weights)
        # We use get_last_hidden_state, NOT the forward() used for pre-training
        out_a = self.bert.get_last_hidden_state(input_ids_a, segment_ids_a)
        out_b = self.bert.get_last_hidden_state(input_ids_b, segment_ids_b)

        # 3. Create Attention Masks (0 for PAD, 1 for Real)
        # Assuming 0 is your PAD token ID
        mask_a = (input_ids_a != 0) 
        mask_b = (input_ids_b != 0)

        # 4. Mean Pooling -> u and v
        u = self.mean_pooling(out_a, mask_a)
        v = self.mean_pooling(out_b, mask_b)

        # 5. Concatenate: (u, v, |u-v|)
        features = torch.cat([u, v, torch.abs(u - v)], dim=1)

        # 6. Classify
        logits = self.classifier(features)
        return logits
    
n_layers = 2    # number of Encoder of Encoder Layer
n_heads  = 2    # number of heads in Multi-Head Attention
d_model  = 256  # Embedding Size
d_ff = 256 * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 16  # dimension of K(=Q), V
n_segments = 2
VOCAB_SIZE = len(word2id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

my_base_bert = BERT(n_layers, n_heads, d_model, d_ff, d_k, n_segments, VOCAB_SIZE, 100, device).to(device)

sbert_model = SentenceBERT(my_base_bert, embed_dim=d_model).to(device)
sbert_model.load_state_dict(torch.load('sbert_model.pt'))
sbert_model.eval()

def clean_text_app(text):
    text = str(text).lower()
    text = re.sub(r"[.,!?\\-]", '', text)
    return text

def get_prediction(premise, hypothesis):
    """
    Runs the model inference and returns probabilities.
    """
    # 1. Tokenize (Reusing your logic)
    def tokenize_text(text):
        tokens = clean_text_app(text).split()
        ids = [word2id.get(w, word2id['[UNK]']) for w in tokens]
        ids = [word2id['[CLS]']] + ids + [word2id['[SEP]']]
        
        # Pad/Truncate to MAX_LEN (e.g., 30)
        max_len = 30
        if len(ids) < max_len:
            ids = ids + [word2id['[PAD]']] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    # 2. Prepare Inputs
    ids_a = tokenize_text(premise)
    ids_b = tokenize_text(hypothesis)
    
    # 3. Predict
    with torch.no_grad():
        logits = sbert_model(ids_a, ids_b)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0] # Convert to numpy array
        
    return probs

# ==========================================
# 2. DASH APP LAYOUT
# ==========================================

app = dash.Dash(__name__)

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '800px', 'margin': '0 auto', 'padding': '20px'}, children=[
    
    # Header
    html.H1("Sentence-BERT NLI Demo", style={'textAlign': 'center', 'color': '#333'}),
    html.P("Enter two sentences to check if they agree, contradict, or are unrelated.", style={'textAlign': 'center', 'color': '#666'}),
    
    html.Hr(),
    
    # Input Area
    html.Div([
        html.Label("Premise (Sentence A):", style={'fontWeight': 'bold'}),
        dcc.Textarea(
            id='input-premise',
            placeholder='E.g., A soccer player is running.',
            style={'width': '100%', 'height': '60px', 'marginBottom': '15px'}
        ),
        
        html.Label("Hypothesis (Sentence B):", style={'fontWeight': 'bold'}),
        dcc.Textarea(
            id='input-hypothesis',
            placeholder='E.g., A person is exercising.',
            style={'width': '100%', 'height': '60px', 'marginBottom': '15px'}
        ),
        
        html.Button('Analyze Similarity', id='btn-predict', n_clicks=0, 
                    style={'width': '100%', 'height': '50px', 'backgroundColor': '#007BFF', 'color': 'white', 'fontSize': '16px', 'border': 'none', 'cursor': 'pointer'}),
    ], style={'backgroundColor': '#f9f9f9', 'padding': '20px', 'borderRadius': '10px'}),
    
    # Output Area
    html.Div(id='output-container', style={'marginTop': '30px'})
])

# ==========================================
# 3. CALLBACKS (INTERACTIVITY)
# ==========================================

@app.callback(
    Output('output-container', 'children'),
    Input('btn-predict', 'n_clicks'),
    State('input-premise', 'value'),
    State('input-hypothesis', 'value')
)
def update_output(n_clicks, premise, hypothesis):
    if n_clicks == 0 or not premise or not hypothesis:
        return html.Div()

    # Get Probabilities
    probs = get_prediction(premise, hypothesis)
    labels = ["Entailment", "Neutral", "Contradiction"]
    colors = ['#28a745', '#ffc107', '#dc3545'] # Green, Yellow, Red
    
    # Determine Winner
    winner_idx = probs.argmax()
    result_text = labels[winner_idx]
    
    # Create Bar Chart
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=probs,
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition='auto',
    )])
    
    fig.update_layout(
        title="Confidence Scores",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        template="plotly_white",
        height=300
    )

    # Return Layout
    return html.Div([
        html.H2(f"Prediction: {result_text}", style={'textAlign': 'center', 'color': colors[winner_idx]}),
        dcc.Graph(figure=fig)
    ])

# ==========================================
# 4. RUN SERVER
# ==========================================
if __name__ == '__main__':
    # If running in Jupyter, use mode='inline'
    # If running as script, remove mode='inline'
    app.run(debug=True,use_reloader=False)