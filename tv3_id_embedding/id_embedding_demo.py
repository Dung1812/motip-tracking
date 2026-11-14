
import torch
import torch.nn as nn


# ==========================================
# 1. XAY DUNG ID DICTIONARY
# ==========================================

class IDDictionary(nn.Module):

    def __init__(self, num_ids=5, embedding_dim=128):
        super(IDDictionary, self).__init__()

        self.num_ids = num_ids
        self.embedding_dim = embedding_dim

        # K ID thong thuong + 1 ID dac biet
        self.newborn_id = num_ids

        # Tao ID Dictionary bang nn.Embedding
        self.id_embedding = nn.Embedding(
            num_embeddings=num_ids + 1,
            embedding_dim=embedding_dim
        )

    # ======================================
    # 2. LAY EMBEDDING THEO ID
    # ======================================

    def get_embedding(self, object_id):

        if object_id < 0 or object_id >= self.num_ids:
            raise ValueError("ID khong hop le!")

        return self.id_embedding(
            torch.tensor(object_id, dtype=torch.long)
        )

    # ======================================
    # 3. EMBEDDING CHO DOI TUONG MOI
    # ======================================

    def get_newborn_embedding(self):

        return self.id_embedding(
            torch.tensor(self.newborn_id, dtype=torch.long)
        )

    # ======================================
    # 4. LAY TOAN BO EMBEDDING
    # ======================================

    def get_all_embeddings(self):

        return self.id_embedding.weight


# ==========================================
# 5. CHUONG TRINH CHINH
# ==========================================

if __name__ == "__main__":

    torch.manual_seed(42)

    K = 5
    C = 128

    # Khoi tao ID Dictionary
    dictionary = IDDictionary(
        num_ids=K,
        embedding_dim=C
    )

    print("========== ID DICTIONARY ==========")

    all_embeddings = dictionary.get_all_embeddings()

    print("Tong so embedding:", all_embeddings.shape[0])
    print("So chieu embedding:", all_embeddings.shape[1])
    print("Shape ID Dictionary:", all_embeddings.shape)

    # ==================================
    # 6. KIEM TRA ID THONG THUONG
    # ==================================

    print("\n========== NORMAL ID EMBEDDING ==========")

    for object_id in range(K):

        embedding = dictionary.get_embedding(object_id)

        print(f"ID {object_id}:")
        print("Embedding shape:", embedding.shape)
        print("5 gia tri dau tien:", embedding[:5])

    # ==================================
    # 7. KIEM TRA NEWBORN EMBEDDING
    # ==================================

    print("\n========== NEWBORN EMBEDDING ==========")

    newborn_embedding = dictionary.get_newborn_embedding()

    print("Newborn ID:", dictionary.newborn_id)
    print("Embedding shape:", newborn_embedding.shape)
    print("5 gia tri dau tien:", newborn_embedding[:5])

    # ==================================
    # 8. KIEM TRA GRADIENT
    # ==================================

    print("\n========== GRADIENT CHECK ==========")

    # Lay embedding cua ID 1, 2 va newborn
    selected_ids = torch.tensor([1, 2, K])

    selected_embeddings = dictionary.id_embedding(selected_ids)

    print("Selected embeddings shape:",
          selected_embeddings.shape)

    # Tao loss de kiem tra gradient
    target = torch.ones_like(selected_embeddings)

    criterion = nn.MSELoss()

    loss = criterion(selected_embeddings, target)

    print("Loss:", loss.item())

    # Lan truyen nguoc
    loss.backward()

    # Kiem tra gradient
    gradients = dictionary.id_embedding.weight.grad

    print("\nGradient shape:", gradients.shape)

    for object_id in selected_ids.tolist():

        grad = gradients[object_id]

        print(f"Gradient cua ID {object_id}:")
        print("Gradient norm:", grad.norm().item())
        print("Co gradient:", grad.abs().sum().item() > 0)

    # Kiem tra mot ID khong duoc su dung
    unused_id = 3

    print(f"\nGradient cua ID {unused_id}:",
          gradients[unused_id].norm().item())

    # ==================================
    # 9. THU CAP NHAT EMBEDDING
    # ==================================

    print("\n========== EMBEDDING UPDATE ==========")

    optimizer = torch.optim.SGD(
        dictionary.parameters(),
        lr=0.1
    )

    old_embedding = dictionary.get_embedding(1).detach().clone()

    optimizer.step()

    new_embedding = dictionary.get_embedding(1).detach()

    print("Embedding truoc khi update:")
    print(old_embedding[:5])

    print("Embedding sau khi update:")
    print(new_embedding[:5])

    print("Embedding da thay doi:",
          not torch.allclose(old_embedding, new_embedding))
