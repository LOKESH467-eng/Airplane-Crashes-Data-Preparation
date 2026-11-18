import torch
from transformers import AutoTokenizer

from model import CONFIG, AviationSLM


class AviationSLMInference:

    def __init__(
        self,
        model_path,
        tokenizer_path,
        device=None
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=False
        )

        if len(self.tokenizer) != CONFIG["vocab_size"]:
            raise ValueError("Tokenizer vocabulary does not match the model.")

        self.model = AviationSLM(CONFIG)

        state_dict = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True
        )

        self.model.load_state_dict(state_dict, strict=True)

        del state_dict

        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.eos_token_id != 151645:
            raise ValueError("Unexpected tokenizer EOS token.")

    @torch.inference_mode()
    def generate_answer(
        self,
        question,
        instruction=(
            "Answer the following aviation question using the "
            "available information. If the information is unavailable, "
            "clearly say that it is not available."
        ),
        max_new_tokens=48
    ):

        prompt = (
            instruction.strip()
            + "\n"
            + question.strip()
            + "\nAnswer:\n"
        )

        token_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False
        )

        if not token_ids:
            raise ValueError("The input prompt is empty.")

        if len(token_ids) >= CONFIG["max_seq_len"]:
            raise ValueError("The prompt is too long.")

        input_ids = torch.tensor(
            token_ids,
            dtype=torch.long
        ).unsqueeze(0).to(self.device)

        generated_tokens = []

        for _ in range(max_new_tokens):

            if input_ids.shape[1] >= CONFIG["max_seq_len"]:
                break

            if self.device.type == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):
                    logits, _ = self.model(input_ids)
            else:
                logits, _ = self.model(input_ids)

            next_token = torch.argmax(
                logits[0, -1, :]
            ).view(1, 1)

            token_id = int(next_token.item())

            if token_id == self.tokenizer.eos_token_id:
                break

            generated_tokens.append(token_id)

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1
            )

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        return answer
