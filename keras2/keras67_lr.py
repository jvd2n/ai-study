from icecream import ic
weight = 0.5
input = 0.5
goal_prediction = 0.8   # 이 값 외엔 다 튜닝 가능
lr = 0.1    # 0.001 # 0.1 / 1 / 0.0001 / 100
epochs = 300

for iteration in range(epochs) :
    prediction = input * weight
    error = (prediction - goal_prediction) ** 2   # mse

    print("Error : " + str(error) + "\tPrediction : " + str(prediction))

    up_prediction = input * (weight + lr)
    up_error = (goal_prediction - up_prediction) ** 2    # mse
    
    down_prediction = input * (weight - lr)
    down_error = (goal_prediction - down_prediction) ** 2
    
    ic(up_prediction, down_prediction)
    ic(up_error, down_error)

    if(down_error <= up_error) :
        weight = weight - lr
    if(down_error > up_error) :
        weight = weight + lr


'''
x_train = 0.5
y_train = 0.8

# weight = 0.5
weight = 0.66
lr = 0.01
epoch = 200

for iteration in range(epoch) :
    y_predict = x_train * weight
    error = (y_predict - y_train) **2   # mse

    print("Error : " + str(error) + "\ty_predict : " + str(y_predict))

    up_y_predict = x_train * (weight + lr)
    up_error = (y_train - up_y_predict) ** 2    # mse
    down_y_predict = x_train * (weight -lr)
    down_error = (y_train - down_y_predict) **2

    if(down_error <= up_error) :
        weight = weight - lr
    if(down_error > up_error) :
        weight = weight +lr
'''