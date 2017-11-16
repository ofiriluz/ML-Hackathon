from Processor import Processor

if __name__ == '__main__':
    processor = Processor(sliding_window_time_frame=30,
                          stddev_threshold=1,
                          risk_iterations=10,
                          minimum_training_size=1000,
                          save_trained_model=True,
                          save_path='./adaboosted_model.pkl')

    # Add the weak learners
    # Run the processor
    processor.start_process()
