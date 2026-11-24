import argparse
import torch
from src.embedding_generator import CustomWebDataset, EmbeddingGeneratorNewWDS, skew_320

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate embeddings for a dataset using SSCD model."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the input parquet file."
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        required=True,
        help="Directory to save the embeddings.",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="models/sscd_disc_large.torchscript.pt",
        help="Path to the pretrained model.",
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=1024,
        help="Size of the generated embeddings.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Batch size for generating embeddings.",
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for computation.")
    args = parser.parse_args()

    # Initialize the embedding model
    embedding_model = lambda: torch.jit.load(args.pretrained_model_path)

    # Create dataset and embedding generator instances
    dataset = CustomWebDataset(df_path=args.dataset)
    embedding_generator = EmbeddingGeneratorNewWDS(
        embedding_dir=args.embedding_dir,
        embedding_size=args.embedding_size,
        embedding_model=embedding_model,
        dataset=dataset,
        cuda=args.cuda,
        batch_size=args.batch_size,
        transform=skew_320,
    )

    # Generate embeddings
    embedding_generator.build_embeddings()
