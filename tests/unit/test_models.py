import pytest
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from text2cypher.finetuning.models.t5_model import T5NoteGenerationModel


@pytest.fixture
def default_t5_model() -> T5NoteGenerationModel:
    return T5NoteGenerationModel(
        model_name="t5-small",
        model_type="t5",
        use_quantization=False,
    )

@pytest.mark.parametrize(
    "model_name, model_type, use_quantization",
    [
        ("t5-small", "t5", False),
        ("t5-base", "t5", False),
        ("t5-nonexistent", "t5", False),
    ]
)
def test_t5_model_initialization(model_name, model_type, use_quantization):
    if model_name == "t5-nonexistent":
        with pytest.raises(Exception):
            T5NoteGenerationModel(
                model_name=model_name,
                model_type=model_type,
                use_quantization=use_quantization
            )
    else:
        model = T5NoteGenerationModel(
            model_name=model_name,
            model_type=model_type,
            use_quantization=use_quantization
        )
        assert model.model_name == model_name
        assert model.model_type == model_type
        assert hasattr(model, "model")
        assert hasattr(model, "tokenizer")



def test_model_generate_note(default_t5_model) -> None:
    conversation = "Doctor: How are you feeling?\nPatient: I have a headache"
    note = default_t5_model.generate_note(conversation, max_length=50)
    assert isinstance(note, str)
    assert len(note) > 0

class DummyDataset(Dataset):
    def __init__(self, tokenizer):
        self.enc = tokenizer("summarize: Hello", return_tensors="pt", padding=True)
    def __len__(self):
        return 2
    def __getitem__(self, idx):
        return {
            "input_ids": self.enc["input_ids"].squeeze(0),
            "attention_mask": self.enc["attention_mask"].squeeze(0),
        }

def test_generate_notes_batch(default_t5_model) -> None:
    dataset = DummyDataset(default_t5_model.tokenizer)
    dataloader = DataLoader(dataset, batch_size=1)

    notes = default_t5_model.generate_notes(dataloader, max_length=20)
    assert isinstance(notes, list)
    assert all(isinstance(n, str) for n in notes)

def test_prefill_and_decode(default_t5_model) -> None:
    conv = "Doctor: Are you sleeping well?\nPatient: Not really"
    encoded = default_t5_model.prefill(conv)

    assert "encoder_outputs" in encoded
    assert hasattr(encoded["encoder_outputs"], "last_hidden_state")
    assert "attention_mask" in encoded

    decoded = default_t5_model.decode(
        encoded["encoder_outputs"],
        encoded["attention_mask"],
        max_length=50
    )
    assert isinstance(decoded, str)
    assert len(decoded) > 0
    assert len(decoded.split()) <= 50

def test_generate_note_with_empty_input(default_t5_model):
    note = default_t5_model.generate_note("", max_length=50)
    assert isinstance(note, str)

@pytest.mark.parametrize("peft_method", ["lora", "prompt_tuning"])
def test_t5_model_with_peft(peft_method):
    model = T5NoteGenerationModel(
        model_name="t5-small",
        model_type="t5",
        peft_method=peft_method,
        use_quantization=False
    )

    assert model.peft_method == peft_method
    assert model.peft_config is not None
    assert hasattr(model.model, "base_model")

    conversation = "Doctor: Do you have any allergies?\nPatient: Just pollen."
    note = model.generate_note(conversation, max_length=50)
    assert isinstance(note, str)
    assert len(note) > 0
