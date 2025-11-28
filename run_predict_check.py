import traceback
from app.predict import load_model, real_time_detect

MODEL_PATH = 'data/spam_fla_model.pkl'

def main():
    try:
        model, compiled = load_model(MODEL_PATH)
        print('Loaded model type:', type(model))
        sample = "Congratulations, you have won a free prize! Click here to claim your cash reward."
        res = real_time_detect(sample, model, compiled)
        print('Result:', res)
    except Exception as e:
        print('Error running prediction:')
        traceback.print_exc()

if __name__ == '__main__':
    main()
