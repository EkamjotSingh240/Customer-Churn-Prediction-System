create database Ecommerce_db;

use Ecommerce_db;

CREATE TABLE ecommerce_cleaned (
    CustomerID INT PRIMARY KEY,
    Churn BOOLEAN,
    Tenure INT,
    PreferredLoginDevice VARCHAR(20),
    CityTier INT,
    WarehouseToHome INT,
    PreferredPaymentMode VARCHAR(20),
    Gender VARCHAR(10),
    HourSpendOnApp INT,
    NumberOfDeviceRegistered INT,
    PreferredOrderCat VARCHAR(30),
    SatisfactionScore INT,
    MaritalStatus VARCHAR(10),
    NumberOfAddress INT,
    Complain BOOLEAN,
    OrderAmountHikeFromlastYear INT,
    CouponUsed INT,
    OrderCount INT,
    DaySinceLastOrder INT,
    CashbackAmount DECIMAL(10,2)
);