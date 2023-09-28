import rpy2.robjects as r_objects

r_objects.r('''
            auto_arima_fit <- function(y_train, frequency) {
                            library(forecast)

                            if (frequency > 0) {
                                y_train = ts(y_train, frequency=frequency)
                            } else {
                                y_train = as.numeric(y_train)
                            }

                            model <- auto.arima(y_train)
                            model
                    }
    ''')

r_objects.r('''
            arima101_fit <- function(y_train, frequency) {
                            library(forecast)

                            if (frequency > 0) {
                                y_train = ts(y_train, frequency=frequency)
                            } else {
                                y_train = as.numeric(y_train)
                            }

                            model <- arima(y_train, order = c(1,0,1))
                            model
                    }
    ''')

r_objects.r('''
                   auto_arima_pred <- function(model,y_test) {
                            library(forecast)

                            y_hat <- fitted(Arima(y_test,model=model))

                    }
                    ''')

r_objects.r('''
                   forecast_fun <- function(model,h) {
                            library(forecast)

                            y_hat <- forecast(model, h=h)$mean

                            return(y_hat)
                    }
                    ''')

r_objects.r('''
                   ets_fit <- function(y_train, frequency) {
                            library(forecast)

                            if (frequency > 0) {
                                y_train = ts(y_train, frequency=frequency)
                            } else {
                                y_train = as.numeric(y_train)
                            }

                            model <- ets(y_train)
                            model

                    }
                    ''')

r_objects.r('''
                   ets_model_predict <- function(model,y_test) {
                            library(forecast)

                            y_hat <- fitted(ets(y_test,model=model))

                    }
                    ''')

r_objects.r('''
                   tbats_fit <- function(y_train, frequency) {
                            library(forecast)

                            if (frequency > 0) {
                                y_train = ts(y_train, frequency=frequency)
                            } else {
                                y_train = as.numeric(y_train)
                            }

                            model <- tbats(y_train)
                            model

                    }
                    ''')

r_objects.r('''
                   tbats_pred <- function(model,y_test) {
                            library(forecast)

                            y_hat <- fitted(tbats(y_test,model=model))

                    }
                    ''')

r_objects.r('''
                   snaive_lazy_forecast <- function(y_train,frequency,h) {
                            library(forecast)

                            y_train = ts(y_train,frequency=frequency)

                            y_hat <- snaive(y_train, h=h)$mean

                            return(y_hat)
                    }
                    ''')

r_objects.r('''
                       theta_forecast <- function(y_train,frequency,h) {
                                library(forecast)

                                y_train = ts(y_train,frequency=frequency)

                                y_hat <- thetaf(y_train, h=h)$mean

                                return(y_hat)
                        }
                        ''')

r_objects.r('''
                   auto_arima_update <- function(model,y_test) {
                            library(forecast)

                            model <- Arima(y_test,model=model)
                            return(model)
                    }
                    ''')

r_objects.r('''
                   ets_update <- function(model,y_test) {
                            library(forecast)

                            suppressMessages(model <- ets(y_test,model=model))
                            return(model)
                    }
                    ''')

r_objects.r('''
                   tbats_update <- function(model,y_test) {
                            library(forecast)

                            model <- tbats(y_test,model=model)
                            return(model)
                    }
                    ''')

tbats_model_fit = r_objects.globalenv['tbats_fit']
tbats_predict = r_objects.globalenv['tbats_pred']
tbats_update = r_objects.globalenv['tbats_update']
ets_model_fit = r_objects.globalenv['ets_fit']
ets_predict = r_objects.globalenv['ets_model_predict']
ets_update = r_objects.globalenv['ets_update']
auto_arima_model_fit = r_objects.globalenv['auto_arima_fit']
auto_arima_predict = r_objects.globalenv['auto_arima_pred']
auto_arima_update = r_objects.globalenv['auto_arima_update']
arima101_model_fit = r_objects.globalenv['arima101_fit']
thetaf_forecast = r_objects.globalenv['theta_forecast']
snaive_forecast = r_objects.globalenv['snaive_lazy_forecast']
model_forecast = r_objects.globalenv['forecast_fun']
