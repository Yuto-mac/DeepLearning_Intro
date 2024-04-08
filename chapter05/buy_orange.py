from layer_naive import MulLayer, AddLayer

apple = 100 
apple_num = 2 
orange = 150 
orange_num = 3 
tax = 1.1

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_app_ora_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple_num, apple)
orange_price = mul_orange_layer.forward(orange_num, orange)
app_ora_price = add_app_ora_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(app_ora_price, tax)

print(price)

# backward
dprice = 1
dapp_ora, dtax = mul_tax_layer.backward(dprice)
dapple, dorange = add_app_ora_layer.backward(dapp_ora)
dapple_price, dapple_num = mul_apple_layer.backward(dapple)
dorange_price, dorange_num = mul_orange_layer.backward(dorange)

print(price) # 715
print(dapple_num, dapple_price, dorange_price, dorange_num, dtax) # 110 2.2 3.3 165 650