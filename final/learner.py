import random

#training data
training_data = [
    (-3,0),
    (-2,0),
    (-1,0),
    (1,1),
    (2,1),
    (3,1)
]

#random strating values
weight= random.uniform(-1,1)
bias= random.uniform(-1,1)

learning_rate= 0.1

for epoch in range(100):
    total_error=0

    for x, correct_answer in training_data:
        #make prediction
        prediction = x*weight+bias
        #calculate errors
        error=correct_answer - prediction
        #adjust the weight and bias
        weight += learning_rate * error * x
        bias += learning_rate * error

        total_error += abs(error)

        print(
            "Epoch: ",
            epoch,
            "Error: ",
            round(total_error, 4)
        )

print("\nFinal weight: ", weight)
print("Final bias: ", bias)