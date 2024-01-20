import numpy as np
import time
import tkinter as tk
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端，你也可以尝试使用其他后端如 'Agg', 'Qt5Agg' 等
import matplotlib.pyplot as plt

# 导入此库的目的是为了对比，而非直接调库求解！
from sklearn.svm import SVC

class SVM:

    def __init__(self, X, y, tol):
        self.X = X  # 训练样本
        self.m, self.n = np.shape(self.X)  # m为训练样本的个数和n为样本的维度
        self.y = y  # 类别
        self.C = 10000  # 正则化常量，用于调整（过）拟合的程度
        self.alphas = np.zeros(self.m)  # 拉格朗日乘子，与样本一一相对
        self.b = 0.0  # 截距b
        self.errors = 0.0 - self.y  # 差值矩阵，用于存储alpha值实际与预测值得差值，其数量与样本一一相对
        self.w = np.zeros(self.n)  # 初始化权重w的值，主要用于线性核函数
        self.tol = tol
        self.satisfied_percent = 0.99 if self.n<4 else 0.95
        np.random.seed(21)


    # 计算所有样本的误差值
    def allE(self):
        return np.dot(self.X, self.w) + self.b - self.y

    # 更新一对乘子
    def updateAlphaPair(self, i, j):
        # 如果相同，则跳过
        if i == j:
            return 0

        # 保留旧值
        alphaIOld = self.alphas[i]
        alphaJOld = self.alphas[j]

        yI = self.y[i]
        yJ = self.y[j]
        s = yI * yJ

        EI = self.errors[i]
        EJ = self.errors[j]


        # 计算alpha的边界
        if (yI != yJ):
            # yI,yJ 异号
            L = max(0, alphaJOld - alphaIOld)
            H = 10000
        elif (yI == yJ):
            # y1,y2
            L = 0
            H = alphaIOld + alphaJOld
        if L == H:
            return 0

        kii = np.dot(self.X[i], self.X[i])
        kjj = np.dot(self.X[j], self.X[j])
        kij = np.dot(self.X[i], self.X[j])

        # 计算 eta，确定乘子更新的方向和步长
        eta = kii + kjj - 2 * kij

        # 如论文中所述，分两种情况根据eta为正positive 还是为负或0来计算计算a2 new
        if (eta > 0):
            # equation (J16) 计算alpha2
            alphaJNew = alphaJOld + yJ * (EI - EJ) / eta
            # clip a2 based on bounds L & H
            # 把a2夹到限定区间 equation （J17）
            alphaJNew = max(alphaJNew,L)
            alphaJNew = min(alphaJNew,H)


        # 如果eta不为正（为负或0）
        else:
            # Equation （J19）
            # 在特殊情况下，eta可能不为正not be positive
            f1 = yI * (EI + self.b) - alphaIOld * kii - s * alphaJOld * kij
            f2 = yJ * (EJ + self.b) - alphaJOld * kjj - s * alphaIOld * kij

            L1 = alphaIOld + s * (alphaJOld - L)
            H1 = alphaIOld + s * (alphaJOld - H)

            Lobj = L1 * f1 + L * f2 + 0.5 * (L1 ** 2) * kii \
                   + 0.5 * (L ** 2) * kjj + s * L * L1 * kij

            Hobj = H1 * f1 + H * f2 + 0.5 * (H1 ** 2) * kii \
                   + 0.5 * (H ** 2) * kjj + s * H * H1 * kij

            if Lobj < Hobj - self.tol:
                alphaJNew = L
            elif Lobj > Hobj + self.tol:
                alphaJNew = H
            else:
                alphaJNew = alphaJOld

        # 超过容差仍不能优化时，跳过
        if (np.abs(alphaJNew - alphaJOld) < self.tol * (alphaJNew + alphaJOld + self.tol )):
            return 0

        # 更新alpha1
        alphaINew = alphaIOld + s * (alphaJOld - alphaJNew)

        # 计算新截距b的值
        b1 = self.b - (EI + yI * (alphaINew - alphaIOld) * kii + yJ * (alphaJNew - alphaJOld) * kij)
        b2 = self.b - (EJ + yI * (alphaINew - alphaIOld) * kij + yJ * (alphaJNew - alphaJOld) * kjj)

        # 更新截距b
        if 0 < alphaINew :
            self.b = b1
        elif 0 < alphaJNew :
            self.b = b2
        else:
            self.b = (b1 + b2) / 2


        # 更新w的值
        self.w = self.w + yI * (alphaINew - alphaIOld) * self.X[i] + yJ * (alphaJNew - alphaJOld) * self.X[j]

        # 更新乘子的值
        self.alphas[i] = alphaINew
        self.alphas[j] = alphaJNew
        # 优化完成，更新差值矩阵的对应值
        self.errors = self.allE()

        return 1

    # 寻找一对准备更新的乘子
    def findAlphaPair(self, j):
        yJ = self.y[j]
        alphaJ = self.alphas[j]
        EJ = self.errors[j]
        rJ = EJ * yJ
        # 重点：这一段的重点在于确定 alpha1, 也就是old a1，优化
        # 下面条件之一满足，进入if开始找第二个alpha，送到take_step进行优化
        if (rJ < -self.tol) or (rJ > self.tol and alphaJ > 0):
            if len(self.alphas[(self.alphas != 0)]) > 1:  # 筛选器
                # 选择Ei矩阵中差值最大的进行优化
                # 要想|E1-E2|最大
                i = np.argmax(np.abs(self.errors - self.errors[j]))
                step_result = self.updateAlphaPair(i, j)
                if step_result:
                    return 1

            # 随机选择
            # 循环所有非0 非C alphas值进行优化，随机选择起始点
            for i in np.where((self.alphas != 0))[0]:
               step_result = self.updateAlphaPair(i, j)
               if step_result:
                   return 1

            # 这里一般是程序的初始阶段
            # 随机选择起始点
            for i in np.arange(self.m):
                step_result = self.updateAlphaPair(i, j)

                if step_result:
                    return 1

        # 如果if条件不满足，说明KKT条件已满足，找其它样本进行优化，则执行下面这句，退出
        return 0

    # 确定超平面
    def fit(self):

        numChanged = 0  # numChanged存放优化返回的结果，如果优化成功，则返回1，反之为0
        examineAll = 1  # examineAll表示从0号元素开始优化，如果所有都优化完成，则赋值为0

        flag = 0
        # 重点：这段的重点在于确定 alpha2，也就是old a 2, 或者说alpha2的下标，old a2和old a1都是heuristically 选择
        while (numChanged > 0) or (examineAll):
            numChanged = 0
            # 全样本遍历
            if examineAll:
                # 从0,1,2,3,...,m顺序选择a2的，送给examine_example选择alpha1，总共m(m-1)种选法
                for i in range(self.m):
                    examine_result = self.findAlphaPair(i)
                    numChanged += examine_result

            # 非边界样本遍历
            else:
                # loop over examples where alphas are not already at their limits
                for i in np.where((self.alphas != 0))[0]:  # 筛选器，用于筛选alpha

                    examine_result = self.findAlphaPair(i)
                    numChanged += examine_result

            if np.mean(self.y * self.errors >= - self.tol) > self.satisfied_percent:
                flag = 1
                break

            if (flag):
                break

            if examineAll == 1:
                examineAll = 0
            elif numChanged == 0:
                examineAll = 1


# 验证原问题的KKT条件是否通过
def check_kkt_conditions(svm_model,tol):
    """
    Check the KKT conditions for the SVM model.
    :param svm_model: Trained SVM model.
    :return: Boolean indicating if KKT conditions are satisfied.
    """
    w = svm_model.w
    b = svm_model.b
    X = svm_model.X
    y = svm_model.y
    alpha = svm_model.alphas
    result=""
    # 1. Original constraints: y_i(w^T x_i + b) >= 1 for all data points
    proportion_satisfied = np.mean([y_i * (np.dot(w, x_i) + b) >= 1 - tol for x_i, y_i in zip(X, y)])
    if(proportion_satisfied>=0.95):
        result+="1.原始限制条件(y_i(w^T x_i + b) >= 1 for all data points) 通过！\n"
        constraints_satisfied=True
    else:
        constraints_satisfied = False
        result+="1.原始限制条件(y_i(w^T x_i + b) >= 1 for all data points) 不通过！\n"


    # 2. Lagrange multipliers: lambda_i >= 0 for all lambda
    lambda_positive_satisfied = all(lambda_i >= 0-tol for lambda_i in alpha)
    if(lambda_positive_satisfied==False):
        result+="2.拉格朗日乘子全部大于等于0 不通过！\n"
    else:
        result+="2.拉格朗日乘子全部大于等于0 通过！\n"

    # 3. Complementary slackness: lambda_i * (1 - y_i(w^T x_i + b)) = 0
    comp_slack_sat = np.mean(np.abs(alpha * (1 - y * np.dot(X, w) + b))<tol)
    if(comp_slack_sat>0.95):
        comp_slack_sat=True
        result += "3.互补松弛条件(lambda_i * (1 - y_i(w^T x_i + b)) = 0) 通过！\n"
    else:
        comp_slack_sat = False
        result += "3.互补松弛条件(lambda_i * (1 - y_i(w^T x_i + b)) = 0) 不通过！\n"

    # 4. Gradient condition:
    # norm(w - sum(lambda_i * y_i * x_i)) < tol
    # sum(lambda_i*y_i)<tol
    grad_cond_w_sat = np.linalg.norm(w - sum(lambda_i * y_i * x_i for x_i, y_i, lambda_i in zip(X, y, alpha))) < tol
    grad_cond_b_sat = sum([lambda_i * y_i for y_i, lambda_i in zip(y, alpha)])<tol
    grad_cond_sat = grad_cond_w_sat&grad_cond_b_sat
    if(grad_cond_sat == False):
        result += "4.梯度等于0 不通过！\n"
    else:
        result+="4.梯度等于0 通过！\n"

    if(all([constraints_satisfied, lambda_positive_satisfied, comp_slack_sat, grad_cond_sat])):
        result+="KKT条件检验结果:True"
    else:
        result+="KKT条件检验结果:False"

    return result


def generate_balanced_data(n, N, w, b):

    # Generate N data points in n dimensions
    X = np.random.randn(int(N),int(n))
    # Initialize labels array
    y = np.zeros(N)
    np.random.seed(125)

    norm_w = np.linalg.norm(w)
    range_min = -norm_w * (1+abs(b))
    range_max = norm_w * (1+abs(b))

    # Half data points with label +1
    for i in range(N // 2):
        while True:
            point = np.random.uniform(range_min,range_max,n)
            # Ensure the point is above the hyperplane
            if np.dot(point, w) + b > range_max:
                X[i] = point
                y[i] = 1
                break

    # Half data points with label -1
    for i in range(N // 2,N):
        while True:
            point = np.random.uniform(range_min,range_max,n)
            # Ensure the point is below the hyperplane
            if np.dot(point, w) + b <-range_max:
                X[i] = point
                y[i] = -1
                break
    return X, y

def run_svm():
    # 获取用户输入
    n = int(n_entry.get())
    N = int(N_var.get())
    w = [float(num) for num in w_entry.get().split(',')]
    b = float(b_entry.get())
    tol = float(tol_entry.get())
    if(n!=len(w)):
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"请重新输入与维度相匹配的w\n")
        return



    scaler = StandardScaler()
    # 使用用户提供的参数生成数据
    startGenerate=time.time()
    X, y = generate_balanced_data(n, N, w, b)
    endGenerate = time.time()
    X =  scaler.fit_transform(X)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"数据生成用时: {endGenerate - startGenerate} 秒\n")


    if(n==2):
        sklearn_svc = SVC(kernel='linear')
        sklearn_svc.fit(X, y)
        # 获取并打印w和b
        w = sklearn_svc.coef_[0]
        b = sklearn_svc.intercept_[0]
        x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        x2 = -(b + w[0] * x1) / w[1]
        plt.plot(x1, x2, color='red', label='Decision Boundary')

    # 创建并训练 SVM 模型
    svm = SVM(X, y, tol)
    start = time.time()
    svm.fit()
    end = time.time()


    if n==2:
        # 绘制数据点
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
        plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class -1')
        w = svm.w
        b = svm.b
        # 绘制决策边界
        x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        x2 = -(b + w[0] * x1) / w[1]
        plt.plot(x1, x2, color='green', label='Decision Boundary')
        plt.show()

    # 输出结果到文本框
    result_text.insert(tk.END, f"SVM求解用时: {end - start} 秒\n")
    result_text.insert(tk.END, f"w: {svm.w}\n b: {svm.b}\n")
    result_text.insert(tk.END, f"原问题的KKT条件检查结果: \n")
    result_text.insert(tk.END, f"{check_kkt_conditions(svm, tol)}\n")

# 创建主窗口
root = tk.Tk()
root.title("SMO算法求解SVM问题")

customFont = ("Arial", 20, "bold")

# 创建并放置控件
tk.Label(root, text="特征空间维度 n:",font=customFont).grid(row=0)
n_var = tk.StringVar()
n_var.set("4")  # 设置默认值为 4
n_entry = tk.Entry(root,width=30,textvariable=n_var)
n_entry.grid(row=0, column=1)

tk.Label(root, text="数据规模 N:",font=customFont).grid(row=1)
N_var = tk.StringVar()
N_var.set("10000")  # 默认值
tk.OptionMenu(root, N_var, "10000", "100000", "1000000").grid(row=1, column=1)

tk.Label(root, text="超平面参数 w (英文逗号分隔):",font=customFont).grid(row=2)
w_var = tk.StringVar()
w_var.set("1,2,4,8")
w_entry = tk.Entry(root,width=30,textvariable = w_var)
w_entry.grid(row=2, column=1)

tk.Label(root, text="超平面截距 b:",font=customFont).grid(row=3)
b_var = tk.StringVar()
b_var.set("4")  # 默认值
b_entry = tk.Entry(root,width=30,textvariable = b_var)
b_entry.grid(row=3, column=1)

tk.Label(root, text="精度 tol(e.g. 1e-6):",font=customFont).grid(row=4)
tol_var = tk.StringVar()
tol_var.set("1e-6")  # 默认值
tol_entry = tk.Entry(root,width=30,textvariable = tol_var)
tol_entry.grid(row=4, column=1)

# 结果输出区域
result_text = tk.Text(root, height=30, width=80,font=customFont)
result_text.grid(row=6, column=0, columnspan=2)

# 运行按钮
run_button = tk.Button(root, text="运行 SVM",font=customFont, command=run_svm)
run_button.grid(row=5, column=0, columnspan=2)

result_text.insert(tk.END, f"当数据规模选择为1e6时,运行程序可能会无响应一段时间，请不要关闭！\n是可以正常求解的，可能需要等一小会儿\n")

# 运行主循环
root.mainloop()


