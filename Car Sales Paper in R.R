setwd('C:/Users/Samuel/Documents/I Know First/Wavelet Practice')
carsales <- read.csv('C:/Users/Samuel/Documents/I Know First/Wavelet Practice/Monthly Car Sales_for replication.csv')
library(reshape)
sales_ts <- ts(carsales, frequency = 12, start = c(1974, 1), end = c(1994, 12))
sales_ts
plot.ts(sales_ts)

mdata <- melt(carsales, id = c('Year'))
mdata['Month'] = 0
month_converter <- function(month){
  match(tolower(month), tolower(month.abb))
}
for (i in 1:252){
  mdata[i,4] = month_converter(mdata[i,2])
}

mdata_sort <- mdata[with(mdata, order(mdata[1], mdata[4])),]
library(zoo)
mdata_sort$tt <- as.yearmon(paste(mdata_sort$Year, mdata_sort$Month, sep = "-"))
mdata_ts <- data.frame(mdata_sort$value)


ts_mdata <- ts(mdata_ts, frequency = 12, start = c(1974, 1), end = c(1994, 12))
xaxis <-  c(seq(from = 1974, to = 1995, by = 1))

plot.ts(ts_mdata, col = c("blue"), ylab = "Car Sales")
title("Monthly Car Sales in Spain")
axis(1, at = 1974:1995, las = 0)

library(astsa)
model1 <- sarima(ts_mdata[1:240],0,1,1,0,1,1,12)
model1$fit

model2 <- sarima(ts_mdata[1:240],2,1,0,0,1,1,12, details = FALSE)
model2$fit

sarima.for(ts_mdata[1:240],12,0,1,1,0,1,1,12)
#fit_sales <- Arima(ts_mdata[1:240], order=c(0,1,1), seasonal=c(0,1,1,12))
#fit_sales2 <- Arima(ts_mdata[241:252], model = fit_sales)
#onestep<- fitted(fit_sales2)

axis.Date(1, as.Date(1974:1995, origin = "1974-01-01"),at = c(seq(1974:1995)))
model1_forecasts <- sarima.for(ts_mdata[1:240],12,0,1,1,0,1,1,12)$pred
model1_se <- sarima.for(ts_mdata[1:240],12,0,1,1,0,1,1,12)$se

U <- model1_forecasts + 2*model1_se
L <- model1_forecasts - 2*model1_se
ts_mdata_U <- append(ts_mdata[1:240], U)
ts_mdata_L <- append(ts_mdata[1:240], L)
ts_mdata_forecasts <- append(ts_mdata[1:240], model1_forecasts)
df <- data.frame(ts_mdata, ts_mdata_forecasts, ts_mdata_U, ts_mdata_L)

ggplot(df, aes(mdata_sort$tt))+
  geom_line(aes(y = ts_mdata_forecasts, color = "Forecasts")) +
  geom_line(aes(y = ts_mdata, color = "Actual Sales"), linetype = "dotted") +
  coord_cartesian(xlim = c(1992, 1995)) +
  labs(title = "Monthly Car Sales and Forecasts with SARIMA Model 1", x = "Date", y = "Car Sales") +
  scale_color_manual(values = c("red", "black"))





errors = model1_forecasts - ts_mdata[241:252]
RMSE = sqrt(mean(errors^2))
#RMSE is 16,963.9 which is the same as in the research paper

####################################################################
####################################################################
####################  Wavelet Decomposition Model ##################
####################  Daubechies Order 8 (db8) Wavelet  ############
####################################################################
####################################################################


#Since we need the number of data points to be a power of 2 and we have 240 points, we will need
#to forecast 16 more(to get a total of 256) using model 1.mo

model1_forecasts16 <- sarima.for(ts_mdata[1:240],16,0,1,1,0,1,1,12)$pred
ts_mdata256 <- append(ts_mdata[1:240], model1_forecasts16)

#Subtract out the mean
ts_centered <- ts_mdata256 - mean(ts_mdata256)


dwt <- wd(ts_centered, filter.number = 8, family = "DaubExPhase", type = "wavelet", bc = "symmetric")
summary(dwt) #levels 8 using db8 wavelet

plot(dwt)

#Extract 0-level Coefficients
C00 <- accessC.wd(dwt, level = 0)
D0 <- accessD.wd(dwt, level = 0)
D1 <- accessD.wd(dwt, level = 1)
D2 <- accessD.wd(dwt, level = 2)
D3 <- accessD.wd(dwt, level = 3)
D4 <- accessD.wd(dwt, level = 4)
D5 <- accessD.wd(dwt, level = 5)
D6 <- accessD.wd(dwt, level = 6)
D7 <- accessD.wd(dwt, level = 7)

#Create Scalogram
E0 <- D0^2
E1 <- sum(D1^2)
E2 <- sum(D2^2)
E3 <- sum(D3^2)
E4 <- sum(D4^2)
E5 <- sum(D5^2)
E6 <- sum(D6^2)
E7 <- sum(D7^2)

#Energy vector to plot scalogram - Note that the 0-level scaling coefficient is NOT included
E <- c(E0/10^9, E1/10^9, E2/10^9, E3/10^9, E4/10^9, E5/10^9, E6/10^9, E7/10^9)
levels = 0:7
plot(levels, E, type = "l", main = "Scalogram of Wavelet Coefficients of x", ylab = "Energies divided by 10^9", xlab = "Levels")

#Because there are two peaks (at level 1 and level 7)
#Split our coefficients vector into d1 and d2 
#d1 takes coefficients from levels 0 to 3
#d2 takes coefficients from levels 4 to 7
#d1 <- c(C00, D0, D1, D2, D3, rep(0, 240))

plot(accessC.wd(dwt, level = 7), type = "l")
plot(accessD.wd(dwt, level = 7), type = "l")
plot(accessC.wd(dwt, level = 6), type = "l")
plot(accessD.wd(dwt, level = 6), type = "l")
plot(accessC.wd(dwt, level = 5), type = "l")
plot(accessC.wd(dwt, level = 4), type = "l")


#c.in <- accessC.wd(dwt, level = 2)
#d.in <- accessD.wd(dwt, level = 2)
#conbar(c.in, d.in, filter.select(filter.number = 8, family = "DaubExPhase"))
#accessC.wd(dwt, level = 3)


#Apply the Inverse Discrete Wavelet Transform separately to d1 and d2
d1 <- nullevels.wd(dwt, levelstonull = c(4,5,6,7))
d2 <- nullevels.wd(dwt, levelstonull = c(0,1,2,3))

#nullevels.wd DOES NOT remove approximation coefficients from d2, so we still need to remove these
d2_fix <- putC.wd(d2, level = 0, v = c(0))

y <- wr(d1) + mean(ts_mdata256) #Long-Term Trend with mean added back in
z <- wr(d2_fix)#Seasonal Component
x <- y + z

#z_centered <- z - mean(z)
#z_c_final <- z_centered[37:240]
y_final <- y[37:240]
z_final <- z[37:240]
x_final <- x[37:240]

df1 <- data.frame(x_final, y_final, z_final)

library(ggplot2)
ggplot(df1, aes(mdata_sort$tt[37:240])) +
  geom_line(aes(y = x_final, color = "x")) + 
  geom_line(aes(y = y_final, color = "y")) +
  geom_line(aes(y = z_final, color = "z")) +
  labs(title = "Series x and its long-term (y) and seasonal (z) components from January 1977 to December 1993", x = "Date", y = "") +
  scale_color_manual(values = c("red", "green", "blue")) +
  theme(axis.text.y = element_text(angle = 45)) +
  scale_y_continuous(breaks = c(-20000, 0, 20000, 40000, 60000, 80000, 100000, 120000), label = c("-20,000", "0", "20,000", "40,000", "60,000", "80,000", "100,000", "120,000"))

#Forecast the y and z components separately
modely_forecast <- sarima(y_final, 1,3,0,0,0,0,0)
modelz_forecast <- sarima(z_final,0,0,0,0,1,1,12)

sarima.for(y_final,12,1,3,0,0,0,0,0)
modely_predictions <- sarima.for(y_final,12,1,3,0,0,0,0,0)$pred

sarima.for(z_final,12,0,0,0,0,1,1,12)
modelz_predictions <- sarima.for(z_final,12,0,0,0,0,1,1,12)$pred

modelx_predictions <- modely_predictions + modelz_predictions

errors_wav <- modelx_predictions - ts_mdata[241:252]
RMSE_wav <- sqrt(mean(errors_wav^2))
#While this RMSE (12,194) is not exactly the same as in the research paper (12,895), 
#it actually performed slightly better which is in turn better than the typical SARIMA forecasts.
ts_mdata_wavforecasts <- append(ts_mdata[1:240], modelx_predictions)
df2 <- data.frame(df, ts_mdata_wavforecasts)
ggplot(df2, aes(mdata_sort$tt))+
  geom_line(aes(y = ts_mdata_forecasts, color = "SARIMA Forecasts")) +
  geom_line(aes(y = ts_mdata, color = "Actual Sales"), linetype = "dotted") +
  geom_line(aes(y = ts_mdata_wavforecasts, color = "Wavelet Forecasts"))+
  coord_cartesian(xlim = c(1992, 1995)) +
  labs(title = "Actual Sales, SARIMA Model 1, Wavelet Model", x = "Date", y = "Car Sales") +
  scale_color_manual(values = c("red", "black", "blue"))


df_errors <- data.frame(abs(errors), abs(errors_wav))


#Least Error by Month
#January: SARIMA 7,375
#February: SARIMA 3,318
#March: Wavlet 10,702
#April: SARIMA 3,515
#May: Wavlet 13,272
#June: Wavlet 16,296
#July: Wavelet 13,934
#August: Wavelet 19,039
#September: Wavelet 13,366
#October: SARIMA 8,022
#November: SARIMA 13,644
#December: Wavelet 10,217


#####################################################################################
################################## Further considerations ###########################
#####################################################################################

#An alternative analysis of series y can be made to provide another global forecast for series x.
#The SARIMA would be (1,4,0)x(1,0,0,12)
modely_alt_forecast <- sarima(y_final,1,4,0,1,0,0,12)
sarima.for(y_final,12,1,4,0,1,0,0,12)
modely_alt_predictions <-sarima.for(y_final,12,1,4,0,1,0,0,12)$pred
modelx_alt_predictions <- modely_alt_predictions + modelz_predictions
errors_wav_alt <- modelx_alt_predictions - ts_mdata[241:252]
RMSE_wav_alt <- sqrt(mean(errors_wav_alt^2))
#This RMSE is 16,275 which is somewhat lower than the original SARIMA forecasts (16,964) but 
#this seems significantly worse than the previous wavelet model.




