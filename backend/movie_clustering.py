import numpy as np
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import logging
import faiss
import pickle
import os

logger = logging.getLogger(__name__)

class MovieEmbeddingNetwork(nn.Module):
    """Neural network for learning movie embeddings with larger dimensions"""
    def __init__(self, input_dim, embedding_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, embedding_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalization

def contrastive_loss(output1, output2, label, margin=0.2):
    """Contrastive loss for similarity learning"""
    dist = F.pairwise_distance(output1, output2)
    loss = (1 - label) * dist.pow(2) + label * torch.clamp(margin - dist, min=0.0).pow(2)
    return loss.mean()

def generate_hard_pairs(movie_embeddings, movie_clusters, n_pairs=1000):
    """Generate pairs for contrastive learning with hard negatives"""
    pairs = []
    labels = []
    cluster_to_movies = defaultdict(list)
    
    # Group movies by cluster
    for movie_id, cluster in enumerate(movie_clusters):
        cluster_to_movies[cluster].append(movie_id)
    
    for _ in range(n_pairs):
        # 50% positive pairs, 50% negative pairs
        if np.random.random() > 0.5:
            # Positive pair (same cluster)
            valid_clusters = [c for c, movies in cluster_to_movies.items() if len(movies) >= 2]
            if not valid_clusters:
                continue
                
            cluster = np.random.choice(valid_clusters)
            id1, id2 = np.random.choice(cluster_to_movies[cluster], 2, replace=False)
            pairs.append((id1, id2))
            labels.append(1)
        else:
            # Negative pair with hard negative mining
            cluster1 = np.random.choice(list(cluster_to_movies.keys()))
            other_clusters = [c for c in cluster_to_movies.keys() if c != cluster1]
            if not other_clusters:
                continue
                
            id1 = np.random.choice(cluster_to_movies[cluster1])
            
            # Find hardest negative
            hardest_negative = None
            smallest_dist = float('inf')
            for cluster2 in other_clusters:
                for id2 in cluster_to_movies[cluster2]:
                    dist = np.linalg.norm(movie_embeddings[id1] - movie_embeddings[id2])
                    if dist < smallest_dist:
                        smallest_dist = dist
                        hardest_negative = id2
            
            pairs.append((id1, hardest_negative))
            labels.append(0)
    
    return pairs, labels

def create_movie_clusters(subtitle_embeddings, review_model_tuple, n_clusters=20):
    """Create movie clusters using combined embeddings and train with contrastive loss"""
    # Unpack review embeddings
    _, review_embeddings = review_model_tuple
    
    # Add debug logging
    logger.info(f"Number of subtitle embeddings: {len(subtitle_embeddings)}")
    logger.info(f"Number of review embeddings: {len(review_embeddings)}")
    
    # Create title to slug mapping from review embeddings
    title_to_slug = {}
    for slug, data in review_embeddings.items():
        if 'metadata' in data and 'title' in data['metadata']:
            title = data['metadata']['title'].lower()
            title_to_slug[title] = slug
    
    logger.info(f"Created mapping for {len(title_to_slug)} movie titles")
    
    # Combine embeddings with normalization
    combined_embeddings = []
    movie_ids = []  # We'll store slugs as IDs
    
    overlap_count = 0
    for subtitle_id, subtitle_data in subtitle_embeddings.items():
        movie_title = subtitle_data['metadata']['title'].lower()
        if movie_title in title_to_slug:
            slug = title_to_slug[movie_title]
            if slug in review_embeddings:
                overlap_count += 1
                try:
                    # Normalize subtitle embedding (mean of chunks)
                    subtitle_emb = normalize(
                        subtitle_data['chunk_embeddings'].mean(axis=0).reshape(1, -1)
                    )
                    # Normalize review embedding
                    review_emb = normalize(
                        review_embeddings[slug]['embedding'].reshape(1, -1)
                    )
                    
                    # Concatenate normalized embeddings
                    combined_emb = np.concatenate([subtitle_emb, review_emb], axis=1).flatten()
                    combined_embeddings.append(combined_emb)
                    movie_ids.append(slug)  # Store the slug as the ID
                    
                    if overlap_count <= 3:  # Log first 3 matches for debugging
                        logger.info(f"Matched movie: '{movie_title}' (slug: {slug})")
                        
                except Exception as e:
                    logger.error(f"Error processing movie '{movie_title}' (slug: {slug}): {e}")
                    continue
    
    logger.info(f"Found {overlap_count} movies with both subtitle and review embeddings")
    logger.info(f"Successfully combined embeddings for {len(combined_embeddings)} movies")
    
    if not combined_embeddings:
        logger.error("No valid combined embeddings found")
        return None
        
    # Convert to numpy array for clustering
    X = np.array(combined_embeddings)
    
    # Initial clustering with KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    # Train network with contrastive loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MovieEmbeddingNetwork(X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    n_epochs = 50
    batch_size = 32
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        # Generate pairs with hard negative mining
        pairs, labels = generate_hard_pairs(X, cluster_labels)
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            
            # Prepare batch data
            embeddings1 = torch.FloatTensor([X[p[0]] for p in batch_pairs]).to(device)
            embeddings2 = torch.FloatTensor([X[p[1]] for p in batch_pairs]).to(device)
            batch_labels = torch.FloatTensor(batch_labels).to(device)
            
            # Forward pass
            output1 = model(embeddings1)
            output2 = model(embeddings2)
            
            # Compute loss
            loss = contrastive_loss(output1, output2, batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(pairs):.4f}")
    
    # Generate final embeddings
    model.eval()
    with torch.no_grad():
        final_embeddings = model(torch.FloatTensor(X).to(device)).cpu().numpy()
    
    # Final clustering with HDBSCAN
    hdbscan = HDBSCAN(min_cluster_size=5, min_samples=2)
    final_clusters = hdbscan.fit_predict(final_embeddings)
    
    # Create Faiss index for fast similarity search
    dimension = final_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(final_embeddings.astype('float32'))
    
    # Create result dictionary
    clustering_results = {
        'movie_ids': movie_ids,
        'embeddings': final_embeddings,
        'clusters': final_clusters,
        'model': model,
        'faiss_index': index
    }
    
    return clustering_results

def find_similar_movies(movie_id, clustering_results, top_k=10):
    """Find similar movies using Faiss for fast nearest neighbor search"""
    try:
        if movie_id not in clustering_results['movie_ids']:
            logger.error(f"Movie ID {movie_id} not found in clustering results")
            return []
        
        idx = clustering_results['movie_ids'].index(movie_id)
        movie_embedding = clustering_results['embeddings'][idx].reshape(1, -1).astype('float32')
        movie_cluster = clustering_results['clusters'][idx]
        
        # Use Faiss for fast similarity search
        D, I = clustering_results['faiss_index'].search(movie_embedding, top_k + 1)
        
        # Format results (skip the first result as it's the query movie itself)
        similarities = []
        for dist, idx in zip(D[0][1:], I[0][1:]):
            other_id = clustering_results['movie_ids'][idx]
            other_cluster = clustering_results['clusters'][idx]
            
            # Convert L2 distance to similarity score (0 to 1 range)
            similarity = 1.0 / (1.0 + float(dist))
            
            # Boost similarity for same cluster
            if other_cluster == movie_cluster and other_cluster != -1:  # -1 is noise cluster
                similarity *= 1.2
            
            # Ensure similarity is between 0 and 1
            similarity = min(max(similarity, 0.0), 1.0)
            
            similarities.append((other_id, similarity))
        
        logger.info(f"Found {len(similarities)} similar movies for {movie_id}")
        return similarities
        
    except Exception as e:
        logger.exception(f"Error finding similar movies for {movie_id}")
        return []

def save_clustering_results(clustering_results, save_path):
    """Save clustering results to disk"""
    # Remove the neural network model before saving as it can cause issues
    results_to_save = {
        'movie_ids': clustering_results['movie_ids'],
        'embeddings': clustering_results['embeddings'],
        'clusters': clustering_results['clusters'],
        'faiss_index': clustering_results['faiss_index']
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(results_to_save, f)
    logger.info(f"Saved clustering results to {save_path}")

def load_clustering_results(load_path):
    """Load clustering results from disk"""
    if not os.path.exists(load_path):
        logger.error(f"No clustering results found at {load_path}")
        return None
        
    with open(load_path, 'rb') as f:
        results = pickle.load(f)
    
    # Recreate NearestNeighbors index
    nn = NearestNeighbors(n_neighbors=11, metric='l2')  # 11 because we want top 10 + self
    nn.fit(results['embeddings'])
    results['nn_index'] = nn
    
    logger.info(f"Loaded clustering results from {load_path}")
    return results 