import numpy as np
from scipy.linalg import eigh
from scipy.special import airy

# Определение модельного потенциала (например, ион-дипольный)
def potential(R, params):
    C = params['C']  # Константа взаимодействия
    return -C / R**2

# Матрица волнового вектора W(R)
def wave_vector_matrix(R, k2_diag, l2_diag, V_matrix):
    return np.diag(k2_diag) - l2_diag(R) - V_matrix(R)

# Диагонализация W(R) на середине интервала
def diagonalize_W(W):
    eigenvalues, eigenvectors = eigh(W)
    return eigenvalues, eigenvectors

# Вычисление параметров для функций Эйри
def compute_airy_params(k2_tilde, W_prime_tilde):
    alpha = -np.cbrt(W_prime_tilde)
    beta = k2_tilde / W_prime_tilde
    return alpha, beta

# Построение пропагатора на основе функций Эйри (упрощенный пример для одного канала)
def airy_propagator(R_n, R_n1, alpha, beta):
    rho_n = R_n - (R_n + R_n1) / 2
    rho_n1 = R_n1 - (R_n + R_n1) / 2
    x_n = alpha * (rho_n + beta)
    x_n1 = alpha * (rho_n1 + beta)
    
    # Функции Эйри и их производные
    Ai_n, Aip_n, Bi_n, Bip_n = airy(x_n)
    Ai_n1, Aip_n1, Bi_n1, Bip_n1 = airy(x_n1)
    
    # Упрощенные вычисления блоков пропагатора
    y1 = Ai_n1 * Bip_n - Bi_n1 * Aip_n
    y2 = Ai_n * Bi_n1 - Bi_n * Ai_n1
    y3 = Aip_n * Bi_n1 - Bip_n * Ai_n1
    y4 = Aip_n1 * Bi_n - Bip_n1 * Ai_n
    
    return y1, y2, y3, y4

# Продвижение матрицы логарифмической производной Y(R)
def propagate_Y(Y_n, y1, y2, y3, y4):
    Y_n1 = y4 - y3 * (Y_n + y1)**-1 * y2  # Упрощенная форма для одного канала
    return Y_n1

# Извлечение S-матрицы (пример для одного канала)
def extract_S_matrix(Y_asymp, k, l):
    # Упрощенная реализация, требует асимптотических функций
    S = (Y_asymp - 1j * k) / (Y_asymp + 1j * k)  # Пример
    return S

# Основная функция для расчета
def airy_propagator_algorithm(R_range, k2_diag, l2_diag, params):
    R = np.linspace(R_range[0], R_range[1], 100)  # Радиальная сетка
    Y = 0  # Начальное условие для Y(R) вблизи R=0
    h = R[1] - R[0]  # Шаг интегрирования
    
    for i in range(len(R) - 1):
        R_n = R[i]
        R_n1 = R[i + 1]
        R_mid = (R_n + R_n1) / 2
        
        # Вычисление W(R) и его диагонализация
        V = potential(R_mid, params)
        W = wave_vector_matrix(R_mid, k2_diag, l2_diag, lambda R: V)
        eigvals, eigvecs = diagonalize_W(W)
        
        # Параметры Эйри для первого канала (упрощение)
        alpha, beta = compute_airy_params(eigvals[0], 1.0)  # W_prime_tilde условно
        
        # Пропагатор
        y1, y2, y3, y4 = airy_propagator(R_n, R_n1, alpha, beta)
        
        # Продвижение Y
        Y = propagate_Y(Y, y1, y2, y3, y4)
    
    # Извлечение S-матрицы в асимптотической области
    S = extract_S_matrix(Y, k2_diag[0], l2_diag(0)[0])
    return S

# Пример использования
if __name__ == "__main__":
    params = {'C': 1.0}  # Параметры потенциала
    R_range = [0.1, 10.0]  # Диапазон R
    k2_diag = [2.0]  # Диагональные элементы k^2
    l2_diag = lambda R: np.diag([0.0])  # Угловая зависимость (упрощение)
    
    S_matrix = airy_propagator_algorithm(R_range, k2_diag, l2_diag, params)
    print("S-матрица:", S_matrix)