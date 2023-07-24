clear
%%
% Cargamos la base de datos
load('mnist.mat');

%%
% Variables de la base de datos
who 

%%
% Accedemos a los datos de entrenamiento

train_images = training.images;  % matriz de las imágenes de entrenamiento
train_labels = training.labels;  % clasificación de las imágenes de entrenamiento

% Accedemos a los datos de prueba

test_images = test.images;       % matriz de las imágenes de prueba
test_labels = test.labels;       % clasificación de las imágenes de prueba

%%
% Tamaño de las muestras

num_train_samples = size(train_images, 3);  % número de imágenes en la muestra de entrenamiento
num_test_samples = size(test_images, 3);    % número de imágenes en la muestra de prueba

disp(['Número de muestras de entrenamiento: ' num2str(num_train_samples)]);
disp(['Número de muestras de prueba: ' num2str(num_test_samples)]);
% "num2str" transforma un numético en un caracter que representa el número 

%%
% Ejemplo de una imagen (concretamente la primera del conjunto de entrnamiento)

first_im = train_images(:, :, 1);  % todas las filas y columnas (:,:) de la 
                                   % primera imagen del conjunto de entrenamiento (1)

first_lab = train_labels(1);       % etiqueta de la primera imagen del conjunto 
                                   % de entrenamiento (el dígito al que corresponde  la imagen)
% fondo negro: 
imshow(first_im); % mostramos la imagen
title(['Dígito ' num2str(first_lab)]); % titulamos la imagen con el dígito correspondiente a su clasificador

% fondo blanco:                                   
imshow(1 - first_im); % mostramos la imagen
title(['Dígito ' num2str(first_lab)]); % titulamos la imagen con el dígito correspondiente a su clasificador

%% 
% Reestructuramos la dimensión de la base de datos 

[nrowtrain, ncoltrain, nimtrain] = size(train_images); % guardamos en la memoria el número de filas (nrowtrain), 
                                                       % el número de columnas (ncoltrain) de la imagen número (nimtrain)

[nrowtest, ncoltest, nimtest] = size(test_images);     % guardamos en la memoria el número de filas (nrowtrain), 
                                                       % el número de columnas (ncoltrain) de la imagen número (nimtrain)

% definimos las nuevas variables de imágenes de prueba y entrenamiento con
% las nuevas dimensiones:
train_images = reshape(train_images, nrowtrain * ncoltrain, nimtrain)'; % reestructuramos los datos de entrenamiento 
                                                                        % a matrices de tamaño 784(=28x28)x60000

test_images = reshape(test_images, nrowtest * ncoltest, nimtest)';      % reestructuramos los datos de prueba 
                                                                        % a matrices de tamaño 784(=28x28)x10000


%%
% Filtramos la base de datos con aquellos dígitos que nos interesen para
% clasificarlos: 1, 7 y 8.

% dígito 1:
train_unos_im = train_images(train_labels == 1,:);    % imágenes de entrenamiento con el dígito 1
train_unos_lab = train_labels(train_labels == 1);     % etiquetas de entrenamiento con el dígito 1
test_unos_im = test_images(test_labels == 1,:);       % imágenes de prueba con el dígito 1
test_unos_lab = test_labels(test_labels == 1);        % etiquetas de prueba con el dígito 1

% dígito 7:
train_sietes_im = train_images(train_labels == 7,:);  % imágenes de entrenamiento con el dígito 7
train_sietes_lab = train_labels(train_labels == 7);   % etiquetas de entrenamiento con el dígito 7
test_sietes_im = test_images(test_labels == 7,:);     % imágenes de prueba con el dígito 7
test_sietes_lab = test_labels(test_labels == 7);      % etiquetas de prueba con el dígito 7
  
% dígito 8
train_ochos_im = train_images(train_labels == 8,:);   % imágenes de entrenamiento con el dígito 8
train_ochos_lab = train_labels(train_labels == 8);    % etiquetas de entrenamiento con el dígito 8
test_ochos_im = test_images(test_labels == 8,:);      % imágenes de prueba con el dígito 8
test_ochos_lab = test_labels(test_labels == 8);       % etiquetas de prueba con el dígito 8

%%
% CLASIFICACIÓN 1-7

% Creamos nuestra base de datos binaria a partir de las filtraciones que hemos hecho anteriormente:

train_images_bin = [train_unos_im;train_sietes_im];    % conjunto de imágenes de entrenamiento (con dígitos 1 y 7)
train_labels_bin = [train_unos_lab;train_sietes_lab];  % conjunto de etiquetas de entrenamiento (con dígitos 1 y 7)
test_images_bin = [test_unos_im;test_sietes_im];       % conjunto de imágenes de prueba (con dígitos 1 y 7)
test_labels_bin = [test_unos_lab;test_sietes_lab];     % conjunto de etiquetas de prueba (con dígitos 1 y 7)

%%
% k-NN: Vamos a clasificar el conjunto de prueba con el algoritmo k-NN 
% para distintos valores de k, concretamente para k = 1, 3, 5, 9. 
% Los elegimos de valor impar para evitar empates entre los clasificadores. 
% Calculamos la precisión de cada una de las soluciones, y comparamos los errores con una gráfica.

kNum = [1,3,5,9];
errors = zeros(length(kNum)); % inicializamos una lista de ceros para después almacenar los errores
for i = 1:length(kNum)
    k = kNum(i);
    model_knn = fitcknn(train_images_bin, train_labels_bin, 'NumNeighbors', k); % fitcknn es el comando de Matlab que crea un modelo del algoritmo k-NN
    predictors_knn = predict(model_knn, test_images_bin); % predecimos la clasificación de las imágenes que sabemos que son un "1"
    accuracy = sum(predictors_knn == test_labels_bin) / numel(test_labels_bin); % precisión del modelo
    fprintf('Precisión en el conjunto de datos de prueba: %.2f%%\n',accuracy * 100); 
    error = (1-accuracy); % calculamos el error del modelo
    errors(i) = error; % almacenamos el error del modelo en la lista de errores
end

% graficar los errores estimados
figure;
plot(kNum, errors, 'o-');
xlabel('Número de vecinos (k)');
ylabel('Error estimado');
title('Error estimado en función del número de vecinos');

% guardamos en la variable error_knn el error más pequeño estimado de los distintos
% modelos k-NN ejecutados. 
error_knn = min(errors);
error_knn = error_knn(1);

%% 
% 1 es el valor de k que menor error ha presentado, por lo tanto obtenemos su matriz de confusión.
% Utilizamos los comandos tic-toc para calcular el tiempo de ejecución.

tic
k = 1;
model_knn = fitcknn(train_images_bin, train_labels_bin, 'NumNeighbors', k);
predictors_knn = predict(model_knn, test_images_bin); % predecimos la clasificación de las imágenes que sabemos que son un "1"
toc

% Matriz de confusión
confusionchart(test_labels_bin,predictors_knn)
title('Matriz de Confusión k-NN 1-7');
xlabel("Predictores")
ylabel("Clasificación real")

%%
% REGRESIÓN LOGÍSTICA

tic
% fitclinear: comando de regresión logística para clasificación binaria
model_rl = fitclinear(train_images_bin, train_labels_bin, 'Learner', 'logistic', 'Regularization', 'ridge');
predictors_rl = predict(model_rl, test_images_bin);
toc

% Precisión
accuracy = sum(predictors_rl == test_labels_bin) / numel(test_labels_bin);
fprintf('Precisión en el conjunto de prueba con regresión logística: %.2f%%\n', accuracy * 100);

% Mtriz de confusión
confusionchart(test_labels_bin, predictors_rl);
title('Matriz de Confusión RL 1-7');
xlabel("Predictores")
ylabel("Clasificación real")

% Error estimado del modelo de regresión logística
error_rl = 1- accuracy;

%%
% PROGRAMACIÓN LINEAL

tic
% Crear el modelo de programación lineal
model_pl = fitcsvm(train_images_bin, train_labels_bin, 'KernelFunction', 'linear', 'Standardize', true);
% Realizar predicciones en el conjunto de prueba
predictors_pl = predict(model_pl, test_images_bin);
toc

% Precisión
accuracy = sum(predictors_pl == test_labels_bin) / numel(test_labels_bin);
fprintf('Precisión en el conjunto de prueba con programación lineal: %.2f%%\n', accuracy * 100);

% Matriz de confusión
confusionchart(test_labels_bin, predictors_pl);
title('Matriz de Confusión PL 1-7');
xlabel("Predictores")
ylabel("Clasificación real")

% Error estimado del modelo de programación lineal
error_pl = 1 - accuracy;

%%
% Graficar los errores estimados
algoritmos = [1,2,3];
errores_algoritmos = [error_knn, error_rl, error_pl];
figure;
scatter(algoritmos, errores_algoritmos);
xticklabels({'k-NN','','','','','RL','','','','','PL'})
xlabel('Algoritmos');
ylabel('Error estimado');
title('Errores estimados 1-7');

%%

% CLASIFICACIÓN 1-8

% Creamos nuestra base de datos binaria a partir de las filtraciones que hemos hecho anteriormente:

train_images_bin = [train_unos_im;train_ochos_im];    % conjunto de imágenes de entrenamiento (con dígitos 1 y 8)
train_labels_bin = [train_unos_lab;train_ochos_lab];  % conjunto de etiquetas de entrenamiento (con dígitos 1 y 8)
test_images_bin = [test_unos_im;test_ochos_im];       % conjunto de imágenes de prueba (con dígitos 1 y 8)
test_labels_bin = [test_unos_lab;test_ochos_lab];     % conjunto de etiquetas de prueba (con dígitos 1 y 8)

%%
% k-NN: Vamos a clasificar el conjunto de prueba con el algoritmo k-NN 
% para distintos valores de k, concretamente para k = 1, 3, 5, 9. 
% Los elegimos de valor impar para evitar empates entre los clasificadores. 
% Calculamos la precisión de cada una de las soluciones, y comparamos los errores con una gráfica.

kNum = [1,3,5,9];
errors = zeros(length(kNum)); % inicializamos una lista de ceros para después almacenar los errores
for i = 1:length(kNum)
    k = kNum(i);
    model_knn = fitcknn(train_images_bin, train_labels_bin, 'NumNeighbors', k); % fitcknn es el comando de Matlab que crea un modelo del algoritmo k-NN
    predictors_knn = predict(model_knn, test_images_bin); % predecimos la clasificación de las imágenes que sabemos que son un "1"
    accuracy = sum(predictors_knn == test_labels_bin) / numel(test_labels_bin); % precisión del modelo
    fprintf('Precisión en el conjunto de datos de prueba: %.2f%%\n',accuracy * 100); 
    error = (1-accuracy); % calculamos el error del modelo
    errors(i) = error; % almacenamos el error del modelo en la lista de errores
end

% graficar los errores estimados
figure;
plot(kNum, errors, 'o-');
xlabel('Número de vecinos (k)');
ylabel('Error estimado');
title('Error estimado en función del número de vecinos');

% guardamos en la variable error_knn el error más pequeño estimado de los distintos
% modelos k-NN ejecutados. 
error_knn = min(errors);
error_knn = error_knn(1);

%% 
% 1 es el valor de k que menor error ha presentado, por lo tanto obtenemos su matriz de confusión.
% Utilizamos los comandos tic-toc para calcular el tiempo de ejecución.

tic
k = 1;
model_knn = fitcknn(train_images_bin, train_labels_bin, 'NumNeighbors', k);
predictors_knn = predict(model_knn, test_images_bin); % predecimos la clasificación de las imágenes que sabemos que son un "1"
toc

% Matriz de confusión
cm_knn = confusionchart(test_labels_bin,predictors_knn);
title('Matriz de Confusión k-NN 1-8');
xlabel("Predictores")
ylabel("Clasificación real")

%%
% REGRESIÓN LOGÍSTICA

tic
% fitclinear: comando de regresión logística para clasificación binaria
model_rl = fitclinear(train_images_bin, train_labels_bin, 'Learner', 'logistic', 'Regularization', 'ridge');
predictors_rl = predict(model_rl, test_images_bin);
toc

% Precisión
accuracy = sum(predictors_rl == test_labels_bin) / numel(test_labels_bin);
fprintf('Precisión en el conjunto de prueba con regresión logística: %.2f%%\n', accuracy * 100);

% Mtriz de confusión
cm_rl = confusionchart(test_labels_bin, predictors_rl);
title('Matriz de Confusión RL 1-8');
xlabel("Predictores")
ylabel("Clasificación real")

% Error estimado del modelo de regresión logística
error_rl = 1- accuracy;

%%
% PROGRAMACIÓN LINEAL

tic
% Crear el modelo de programación lineal
model_pl = fitcsvm(train_images_bin, train_labels_bin, 'KernelFunction', 'linear', 'Standardize', true);
% Realizar predicciones en el conjunto de prueba
predictors_pl = predict(model_pl, test_images_bin);
toc

% Precisión
accuracy = sum(predictors_pl == test_labels_bin) / numel(test_labels_bin);
fprintf('Precisión en el conjunto de prueba con programación lineal: %.2f%%\n', accuracy * 100);

% Matriz de confusión
cm_pl = confusionchart(test_labels_bin, predictors_pl);
title('Matriz de Confusión PL 1-8');
xlabel("Predictores")
ylabel("Clasificación real")

% Error estimado del modelo de programación lineal
error_pl = 1 - accuracy;

%%
% Graficar los errores estimados
algoritmos = [1,2,3];
errores_algoritmos = [error_knn, error_rl, error_pl];
figure;
scatter(algoritmos, errores_algoritmos);
xticklabels({'k-NN','','','','','RL','','','','','PL'})
xlabel('Algoritmos');
ylabel('Error estimado');
title('Errores estimados 1-8');

